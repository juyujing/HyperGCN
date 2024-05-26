import dgl
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import dgl.function as fn
import numpy as np


def load_graph(user_num) -> list:
    # load different view graph
    records = []
    f = open("data/beibei/train.txt", "r")
    for line in f:
        record = line.strip().split("\t")
        records.append(record)
    init_item_init = []
    init_item_item = []
    init_part_init = []
    init_part_part = []
    part_item_part = []
    part_item_item = []
    for record in records:
        init_item_init.append(int(record[0]))
        init_item_item.append(int(record[1]) + user_num)
        for p in record[2:]:
            init_part_init.append(int(record[0]))
            init_part_part.append(int(p))
            part_item_part.append(int(p))
            part_item_item.append(int(record[1]) + user_num)

    records = []
    f = open("data/beibei/valid.txt", "r")
    for line in f:
        record = line.strip().split("\t")
        records.append(record)
    for record in records:
        init_item_init.append(int(record[0]))
        init_item_item.append(int(record[1]) + user_num)

    init_item_graph = dgl.graph((np.array(init_item_init), np.array(init_item_item)))
    init_part_graph = dgl.graph((np.array(init_part_init), np.array(init_part_part)))
    part_item_graph = dgl.graph((np.array(part_item_part), np.array(part_item_item)))

    # GCN is bidirectional
    g_edges = init_item_graph.edges()
    init_item_graph.add_edges(g_edges[1], g_edges[0])
    g_edges = init_part_graph.edges()
    init_part_graph.add_edges(g_edges[1], g_edges[0])
    g_edges = part_item_graph.edges()
    part_item_graph.add_edges(g_edges[1], g_edges[0])

    return [init_item_graph, init_part_graph, part_item_graph]

def gcn_msg(edges):
        return {'m':edges.src['h']}

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        init.xavier_normal_(self.linear.weight)

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.gcn_msg = gcn_msg # fn.copy_u(src='h', out='m')
        self.gcn_reduce = fn.mean(msg='m', out='h')
    
    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(self.gcn_msg, self.gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata['h']


class GraphGCN(nn.Module):
    def __init__(self, in_dimension, hidden_dimension, out_dimension):
        super(GraphGCN, self).__init__()
        self.gcn1 = GCN(in_dimension, hidden_dimension, F.relu)
        self.gcn2 = GCN(hidden_dimension, out_dimension, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x