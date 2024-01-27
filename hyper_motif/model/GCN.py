import torch
import torch.nn as nn
import dgl

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.aggregation_weight = nn.Parameter(torch.tensor(0.0))
        self.activation = activation
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, node):
        h_self = node.data['h']
        h_neigh = node.data['h_neigh']
        alpha = torch.sigmoid(self.aggregation_weight)
        aggregated_h = alpha * h_self + (1 - alpha) * h_neigh
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h':h}


def gcn_msg(edges):
    return {'m':edges.src['h']}

def gcn_reduce(nodes):
    h_neigh = torch.mean(nodes.mailbox['m'],dim=1)
    return {'h_neigh':h_neigh}

def gat_msg(edges):
    return {'m':edges.src['h'], 'a':edges.src['a']}

def gat_reduce(nodes):
    alpha = torch.softmax(nodes.mailbox['a'],dim=1)
    h_neigh = torch.sum(alpha*nodes.mailbox['m'],dim=1)
    return {'h_neigh':h_neigh}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN,self).__init__()
        self.apply_mod = NodeApplyModule(in_feats,out_feats,activation=activation)
        self.gcn_msg = gcn_msg
        self.gcn_reduce = gcn_reduce
        # self.attention = nn.linear(in_feats, 1)
        
    def forward(self, g, feats):
        g.ndata['h'] = feats
        # g.ndata['a'] = self.attention(feats)
        g.update_all(self.gcn_msg, self.gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata['h']
    

class GCNNet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, activation):
        super(GCNNet, self).__init__()
        self.gcn1 = GCN(in_feats=in_feats,out_feats=hidden_feats,activation=activation)
        self.gcn2 = GCN(in_feats=hidden_feats,out_feats=out_feats,activation=activation)
    
    def forward(self, g, feats):
        out = self.gcn1(g, feats)
        out = self.gcn2(g,out)
        return out

