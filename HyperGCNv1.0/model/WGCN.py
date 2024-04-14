import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

# 创建一个带权图
src = torch.tensor([0, 0, 3, 1, 2, 4])
dst = torch.tensor([1, 2, 4, 0, 0, 3])

g = dgl.graph((src, dst),num_nodes=5)

g.edata['w'] = torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.float32)

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
    def __init__(self, in_feats, out_feats,activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.aggregation_weight = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_normal_(self.linear.weight)
        
    def forward(self, nodes):
        h_self = nodes.data['h']  # 节点本身的特征
        h_neigh = nodes.data['h_neigh']  # 邻居节点的特征
        alpha = torch.sigmoid(self.aggregation_weight)  # 使用Sigmoid函数将参数值限制在0到1之间
        aggregated_h = alpha * h_self + (1 - alpha) * h_neigh
        h = self.linear(aggregated_h)
        h = self.activation(h)
        # h = h_neigh
        return {'h': h}

# 创建带权图卷积层
class WeightedGCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(WeightedGCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats,out_feats,activation)
        self.msg_func = msg_func
        self.reduce_func = reduce_func
        
    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            g.update_all(msg_func, reduce_func)
            g.apply_nodes(func=self.apply_mod)
            return g.ndata.pop('h')

# 创建带权图卷积层
conv = WeightedGCN(2, 2, F.relu)

# 输入特征
features = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.3, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=torch.float32)

# 执行带权图卷积
output = conv(g, features)
print(output)
