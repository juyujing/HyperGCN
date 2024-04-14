import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

def load_hyper_graph(user_num,item_num) -> list:
    # load hyper graph
    i = 0
    records = []
    index = []
    f = open("data/train.txt", "r")
    for line in f:
        record = line.strip().split("\t")
        int_record = [int(x) for x in record]
        int_record[1] += user_num
        temp = []
        temp += [i]*len(int_record)
        records.append(int_record)
        index.append(temp)
        i += 1
    
    f = open("data/valid.txt", "r")
    for line in f:
        record = line.strip().split("\t")
        int_record = [int(x) for x in record]
        int_record[1] += user_num
        temp = []
        temp += [i]*len(int_record)
        records.append(int_record)
        index.append(temp)
        i += 1
    
    flatten_records = [item for sublist in records for item in sublist]
    flatten_index = [item for sublist in index for item in sublist]
    
    indices = torch.LongTensor([flatten_records, flatten_index])
    values = torch.FloatTensor([1] * len(flatten_records))
    H = torch.sparse.FloatTensor(indices, values, torch.Size([user_num+item_num, i]))
    
    return H

class HyperGCN(nn.Module):
    def __init__(self, in_feats, out_feats,device):
        super(HyperGCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_edges = 362337+164981
        self.device = device
        self.W_ = torch.ones(self.num_edges, 1).to(device)
        # self.W_ = nn.Linear(self.num_edges, self.num_edges)
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = nn.Sigmoid()
        
        # init.constant_(self.W_.weight, 1.0)

    def forward(self,H,feats):
        D_v = (torch.sparse.mm(H,self.W_)).to_dense().pow(-0.5).view(-1).to(self.device)
        size = D_v.shape[0]
        indices = torch.stack([torch.arange(size), torch.arange(size)]).to(self.device)
        D_v_2 = torch.sparse_coo_tensor(indices, D_v, (size, size))
        
        D_e = (torch.sparse.sum(H,dim=0)).to_dense().pow(-1.0).view(-1).to(self.device)
        indices = torch.stack([torch.arange(self.num_edges), torch.arange(self.num_edges)]).to(self.device)
        D_e_ = torch.sparse_coo_tensor(indices, D_e, (self.num_edges, self.num_edges))
        
        W = torch.sparse_coo_tensor(indices, self.W_.view(-1),(self.num_edges, self.num_edges))
        HW = torch.sparse.mm(H, W)
        HWD_e_ = torch.sparse.mm(HW,D_e_)
        HWD_e_H_T = torch.sparse.mm(HWD_e_,H.transpose(0,1))
        A = torch.sparse.mm(torch.sparse.mm(D_v_2,HWD_e_H_T),D_v_2)
        
        Y = self.activation(self.linear(torch.sparse.mm(A,feats)))
        
        return Y

if __name__ == '__main__':
    device = torch.device('cuda:1')
    HyperGraph = load_hyper_graph(125012,30516).to(device)
    embed = nn.Embedding(125012+30516, 256)
    model = HyperGCN(256,128,device).to(device)
    x = model(HyperGraph, embed(torch.arange(0, 125012+30516)).to(device))
    print(x)