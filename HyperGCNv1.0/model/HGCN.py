import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import GCN
import Loader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

def has_nan(tensor):
    return torch.isnan(tensor).any().item()

def all_nan(tensor):
    return torch.isnan(tensor).all().item()

# 定义一个 hook 函数，用于打印中间梯度信息
def print_grad(name):
    def hook(grad):
        print(f"Gradient for parameter {name}:")
        print(grad)
    return hook

def print_gradients(model):
    # 在每个层注册 hook
    hooks = []
    for name, param in model.named_parameters():
        hook = param.register_hook(print_grad(name))
        hooks.append(hook)
    return hooks

# 计算 BPR Loss
def single_bpr_loss(pos_score, neg_scores):
    loss = -F.logsigmoid(pos_score - neg_scores)
    loss = torch.mean(loss)
    # 返回平均 BPR 
    return loss

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_elem):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.calculate_scores = nn.Linear(num_elem,1)
        self.calculate_weights = nn.Softmax(dim=1)  # Apply softmax along the last dimension
        self.final_out = nn.Linear(hidden_size, input_size)

    def forward(self, input, mask, num_groups):
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)
        mask = torch.BoolTensor((mask == 0).T) # Notate the zero element
        scores = self.calculate_scores(torch.matmul(query, key.t())).view(1,-1).repeat(num_groups,1)
        scores[mask] = float('-inf') # Replace the corresponding position with -inf
        row_all_inf_mask = torch.all(scores == -float('inf'), dim=1)
        scores[row_all_inf_mask, :] = -1e38
        attention_weights = self.calculate_weights(scores)
        # attention_weights[row_all_inf_mask, :] = 0.0
        aggregation = torch.mm(attention_weights, value)
        return aggregation, attention_weights

# 定义消息函数和聚合函数
def msg_func(edges):
    m = torch.multiply(edges.data['w'].view(-1,1).expand(-1,edges.src['h'].shape[1]),edges.src['h'])
    w_m = edges.data['w'].view(-1,1).expand(-1,edges.src['h'].shape[1])
    return {'m': m, 'w_m': w_m}

def reduce_func(nodes):
    sum_w = torch.sum(nodes.mailbox['w_m'],dim=1)
    h_neigh = torch.sum(nodes.mailbox['m'], dim=1)
    return {'h_neigh': h_neigh/sum_w}

# 定义NodeApplyModule
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, nodes):
        h_self = nodes.data['h']  # 节点本身的特征
        h_neigh = nodes.data['h_neigh']  # 邻居节点的特征
        alpha = 0.8
        aggregated_h = alpha * h_self + (1 - alpha) * h_neigh
        h = self.linear(aggregated_h)
        h = self.activation(h)
        # h = h_neigh
        return {'h': h}

# 创建带权图卷积层
class WeightedGCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(WeightedGCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats,out_feats)
        self.msg_func = msg_func
        self.reduce_func = reduce_func
        
    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            g.update_all(msg_func, reduce_func)
            g.apply_nodes(func=self.apply_mod)
            return g.ndata.pop('h')


class HGCN(nn.Module):
    def __init__(self, num_users, num_items, in_dim, out_dim, attention_hidden_dim, device): # 3 2 2 128 32 64 ReLU
        super(HGCN, self).__init__()
        self.num_users = num_users # 3
        self.num_items = num_items # 2
        self.in_dim = in_dim # 128
        self.out_dim = out_dim # 32
        # self.Aggregator = SelfAttention(self.in_dim,attention_hidden_dim,self.num_users+self.num_items) # 128 64 3+2 2
        self.userAggregator = SelfAttention(self.in_dim,attention_hidden_dim,self.num_users) # 128 64 3 2
        self.itemAggregator = SelfAttention(self.in_dim,attention_hidden_dim,self.num_items) # 128 64 2 2
        self.device = device
        self.gate_init_p = nn.Sequential(
            nn.Linear(in_dim*2,1),
            nn.Sigmoid()
        ) # 128*2 Sigmoid 得到的是权重
        self.weights_trans = nn.Sequential(
            nn.Linear(self.num_users+self.num_items,self.num_users+self.num_items)
        ) # 3+2 3+2
        self.WGCN = WeightedGCN(attention_hidden_dim*2, out_dim) # 64 32 ReLU
        self.compute_score = nn.Sequential(
            nn.Linear(out_dim,1),
            nn.ReLU()
        )
        self.trans_feats = nn.Linear(in_dim,out_dim)
        

    def forward(self, H, feats,num_groups, init=None):
        if init!=None:
            group_feats_init = feats.weight[init]

        users_feats = feats.weight[:self.num_users] # 3*128
        items_feats = feats.weight[self.num_users:] # 2*128

        users_mask = H[0:self.num_users] # 3*2
        items_mask = H[self.num_users:] # 2*2

        group_feats_p,group_u_weights = self.userAggregator(users_feats,users_mask,num_groups=num_groups) # torch.Size([2, 64]) 
        group_feats_i,group_i_weights = self.itemAggregator(items_feats,items_mask,num_groups=num_groups) # torch.Size([2, 64])

        group_feats_i = torch.where(torch.isnan(group_feats_i), torch.tensor(0.0).to(self.device), group_feats_i)
        group_i_weights = torch.where(torch.isnan(group_i_weights), torch.tensor(0.0).to(self.device), group_i_weights)

        group_weights = torch.cat((group_u_weights,group_i_weights),dim=1) # 2*(3+2) torch.Size([2, 5])
        group_feats = torch.cat((group_feats_p,group_feats_i),dim=1) # 2*128 torch.Size([2, 128])

        if init!=None:
            alpha = self.gate_init_p(torch.cat((group_feats_init,group_feats_p),dim=0)).view(-1,1) # [num_groups,1]
            group_feats_u = torch.multiply((1.0-alpha).expand(-1,self.in_dim),group_feats_init)+torch.multiply(alpha.expand(-1,self.in_dim),group_feats_p) # group=(1-alpha)*init+alpha*parti
        else:
            group_feats_u = group_feats_p

        
        group_feats = torch.cat((group_feats_u,group_feats_i),dim=1)

        # Similarity Matrix: Calculate the similarity between hyper graphs
        if init==None: # init absent

            # 群组权重矩阵本身与其转置相乘，计算多维向量相似度
            group_weights = group_weights.to_dense()

            weights_product = group_weights.mm(group_weights.t()) # torch.Size([2, 2])

            row_squared_norms = torch.sum(group_weights.square(),dim=1)
            L2= row_squared_norms.sqrt()
            L2 = L2.to_dense().unsqueeze(0).repeat(num_groups,1)
            L2 = L2*(L2.t())
            epsilon = 1e-8
            L2 = torch.reciprocal(L2 + epsilon)

            simi__mat_indices = weights_product.nonzero()
            L2_line = L2[simi__mat_indices[0, :], simi__mat_indices[1, :]]
            simi_mat_weights =  weights_product[simi__mat_indices[0, :], simi__mat_indices[1, :]].mul(L2_line) # simi_mat_weights =  weights_product._values()*L2_line

            edges_src = simi__mat_indices[0, :]
            edges_dst = simi__mat_indices[1, :]
            weights = simi_mat_weights
        else: # init exist
            pass
        
        g = dgl.graph((edges_src, edges_dst),num_nodes=num_groups)
        g.edata['w'] = weights

        hyperconv_feats = self.WGCN(g,group_feats)

        if self.training:
            total_loss = 0
            feats_32 = self.trans_feats(feats.weight)
            i_g_pairs = np.column_stack(np.where(items_mask == 1))
            i_g_pairs[:, 0] += self.num_users
            
            for row in i_g_pairs:
                group_embedding = hyperconv_feats[row[1]]
                pos_item_embedding = feats_32[row[0]]
                neg_item_embedding = feats_32[np.random.choice(np.arange(self.num_users, self.num_users + self.num_items), size=10, replace=False)]
                pos_score = self.compute_score(group_embedding * pos_item_embedding)
                neg_scores = self.compute_score(group_embedding * neg_item_embedding)
                total_loss += single_bpr_loss(pos_score, neg_scores)
            loss = total_loss/(i_g_pairs.shape[0])
            return loss
        else:
            feats_32 = self.trans_feats(feats.weight)

            return loss,hyperconv_feats
        


if __name__ == '__main__':
    device_id = "cuda:"+str(3)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    if 1:
        batch_size = 4000
        H = np.zeros((8643+25081,22733))
        E = []
        file_path = 'hyper_motif/data/Weeplaces/group_user.txt'  # Document path
        with open(file_path, 'r') as file:
            for line in file:
                group, member = map(int,map(float, line.strip().split(',')))
                H[member][group] = 1
        file_path = 'hyper_motif/data/Weeplaces/group_item_train.txt'
        with open(file_path, 'r') as file:
            for line in file:
                group, item = map(int,line.strip().split(','))
                H[item+8643][group] = 1
        
        '''
        HGCN
        '''
        feats = nn.Embedding(8643+25081,128).to(device)
        
        model_a = HGCN(num_users=8643, num_items=25081, in_dim=128,out_dim=32, attention_hidden_dim=64, device=device)
        model_a.to(device).train()
        print(model_a)
        
        optimizer = optim.Adam(model_a.parameters(), lr=0.005, weight_decay=1e-4)  # 适当的 weight_decay 值
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 示例调度器

        # 训练

        # 创建 DataLoader 实例
        dataset = Loader.MyDataset(H,batch_size=batch_size)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
        
        num_epochs = 200
        for epoch in range(num_epochs):
            total_loss = 0.0
            for H_b in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
                optimizer.zero_grad()
                # hooks = print_gradients(model_a)

                # 计算损失
                loss_a = model_a(H_b.numpy(),feats,num_groups=H_b.shape[1])
                with open('result.txt', 'a') as file:
                    file.write(str(loss_a)+'\n')
                print('loss_a: ', loss_a)

                # 反向传播
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=1.0)  # 适当的 max_norm 值
                
                # 更新模型参数
                optimizer.step()
                # for hook in hooks:
                #     hook.remove()
                torch.cuda.empty_cache()
                total_loss += loss_a.item()

            scheduler.step()
            # print_gradients(model_a)
            average_loss = total_loss / len(dataloader)
            print(f'\nEpoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}\n')

        # 保存模型的状态字典（仅包含模型参数）
        torch.save(model_a.state_dict(), 'hyper_motif/dict/WGCN.pth')

        # 加载保存的模型参数
        model_a.load_state_dict(torch.load('hyper_motif/dict/WGCN.pth'))

        # 测试
        model_a.eval()
        with torch.no_grad():
            
            pass


        '''
        GCN
        '''
        # model_b = GCN.GCNNet(in_feats=128,hidden_feats=64,out_feats=32)



    else:
        model = HGCN(num_users=3, num_items=2, num_groups=2, in_dim=128,out_dim=32, attention_hidden_dim=64, device=device)
        model.train()
        feats = nn.Embedding(5,128)
        H = np.array([[1,0],
                    [1,1], 
                    [0,1],
                    [1,1],
                    [0,1]])
        out = model(H,feats)
        print(out)

