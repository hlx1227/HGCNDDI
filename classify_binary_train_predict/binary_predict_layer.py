import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
import torch
import numpy as np


# 异质图卷积模型
class HGCNLayer(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        # self.sequential1 = nn.Sequential(
        #     nn.Linear(80, 1024),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(1024),
        #     # output layer
        #     nn.Linear(1024, 80),
        #     nn.ReLU(True),
        # )
        # self.sequential2 = nn.Sequential(
        #     nn.Linear(4, 128),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(128),
        #     # output layer
        #     nn.Linear(128, 4),
        #     nn.ReLU(True),
        # )
        self.sequential1 = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            # output layer
            nn.Linear(128, 100),
            nn.ReLU(True),
        )
        self.sequential2 = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            # output layer
            nn.Linear(64, 4),
            nn.ReLU(True),
        )

    def forward(self, graph, inputs):
        # h = self.conv1(graph, d1_cat_fts)
        h = self.conv1(graph, inputs)
        # h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.sequential1(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: self.sequential2(v) for k, v in h.items()}
        return h


class SAGELayer(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # self.conv1 = dglnn.HeteroGraphConv({
        #     rel_names[0]: dglnn.SAGEConv(in_feats, hid_feats, 'mean'),
        #     rel_names[1]: dglnn.GraphConv(in_feats, hid_feats),
        #     rel_names[2]: dglnn.GATConv(in_feats, hid_feats, num_heads=5),
        #     rel_names[3]: dglnn.GraphConv(in_feats, hid_feats),
        # }, aggregate='sum')
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats, 'mean')
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats, out_feats, 'mean')
            for rel in rel_names}, aggregate='sum')
        self.w1 = nn.Linear(1939, 20),

        # self.conv1 = dglnn.SAGEConv(
        #     in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        # self.conv2 = dglnn.SAGEConv(
        #     in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        # self.linear = torch.nn.Linear(hid_feats, out_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: self.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}
        # h = {k: self.w1(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: self.relu(v) for k, v in h.items()}
        return h


class GATLayer(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel_names[0]: dglnn.SAGEConv(in_feats, hid_feats, 'mean'),
            rel_names[1]: dglnn.GraphConv(in_feats, hid_feats),
            rel_names[2]: dglnn.EdgeConv(in_feats, hid_feats),
            rel_names[3]: dglnn.EdgeConv(in_feats, hid_feats),
        }, aggregate='sum')
        self.conv2 = dglnn.GATConv(hid_feats, out_feats, num_heads=3)
        # self.linear = torch.nn.Linear(hid_feats, out_feats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: self.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: self.relu(v) for k, v in h.items()}
        return h
