import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from binary_predict_layer import HGCNLayer, SAGELayer, GATLayer


class Model(nn.Module):
    def __init__(self, in_features2, in_features3, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage2 = HGCNLayer(in_features2, hidden_features, out_features, rel_names)
        self.sage3 = HGCNLayer(in_features3, hidden_features, out_features, rel_names)
        # self.sage1 = SAGELayer(in_features1, hidden_features, out_features, rel_names)
        # self.sage2 = SAGELayer(in_features2, hidden_features, out_features, rel_names)
        # self.sage3 = SAGELayer(in_features3, hidden_features, out_features, rel_names)
        self.pred = HeteroMLPPredictor(out_features, len(rel_names))
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

    def forward(self, g, node2_features, mpnn_features, dec_graph):
        h2 = self.sage2(g, node2_features)
        h3 = self.sage3(g, mpnn_features)
        h = self.pred(dec_graph, h2, h3, mpnn_features)
        # h = self.relu(h)
        # h = self.dropout(h)
        return h


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W1 = nn.Linear(in_dims * 4, 256)  # concat
        # self.W1 = nn.Linear(in_dims * 2, 256)  # sum
        # self.W1 = nn.Linear(in_dims, 256)
        self.W2 = nn.Linear(256, n_classes)
        # self.W1 = nn.Linear(in_dims * 2, n_classes)
        self.relu = nn.ReLU()

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        # y = self.W1(x)
        y = self.W2(self.W1(x))
        # y = F.leaky_relu(y)
        return {'score': y}

    def forward(self, graph, h2, h3, h4):
        # concat
        # features_concat = torch.cat([list(h1.values())[0], list(h2.values())[0], list(h3.values())[0], list(h4.values())[0]], dim=1) #包括mpnn
        features_concat = torch.cat([list(h2.values())[0], list(h3.values())[0]],
                                    dim=1)  # （sim  mpnn&hgcn   RW）
        # features_concat = torch.cat([list(h2.values())[0], list(h3.values())[0]], dim=1)  # （ mpnn&hgcn   RW）
        # sum
        # features_sum = list(h1.values())[0] + list(h2.values())[0]
        # drug_features_sum = {'drug': features_sum, 'drug': features_sum}

        drug_features_concat = {'drug': features_concat, 'drug': features_concat}
        # h是对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = list(drug_features_concat.values())[0]  # 一次性为所有节点类型的 'h'赋值
            # graph.ndata['h'] = list(h2.values())[0]  # 一次性为所有节点类型的 'h'赋值
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# class EncoderLayer(torch.nn.Module):
#     def __init__(self, input_dim, n_heads):
#         super(EncoderLayer, self).__init__()
#         self.attn = MultiHeadAttention(input_dim, n_heads)
#         self.AN1 = torch.nn.LayerNorm(input_dim)
#
#         self.l1 = torch.nn.Linear(input_dim, input_dim)
#         self.AN2 = torch.nn.LayerNorm(input_dim)
#
#     def forward(self, X):
#         output = self.attn(X)
#         X = self.AN1(output + X)
#
#         output = self.l1(X)
#         X = self.AN2(output + X)
#
#         return X

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads, ouput_dim)
        self.AN1 = torch.nn.LayerNorm(ouput_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output)

        return X
