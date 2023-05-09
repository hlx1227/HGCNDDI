import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import random


def calculate_pos_sample_number(graph_csv):
    level = graph_csv.Level.tolist()
    pos_num = 0
    neg_num = 0
    srcs = []
    dsts = []
    for i, j in enumerate(level):
        if j != 'Unknown':
            pos_num += 1
        if j == 'Unknown':
            neg_num += 1
            # a = graph_csv.iloc[i, 0]
            # b = graph_csv.iloc[i, 1]
            srcs.append(graph_csv.iloc[i, 0])
            dsts.append(graph_csv.iloc[i, 1])

    return pos_num, neg_num, srcs, dsts


def neg_sample(graph, pos_num, neg_num):
    num_nodes = graph.num_nodes()
    srcs = []
    dsts = []
    neg_number = 0
    flag = True

    for i in range(num_nodes):
        for j in range(num_nodes):
            # 随机采样
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if graph.has_edges_between(src, dst) is False:
                neg_number += 1
                srcs.append(src)
                dsts.append(dst)
                print(neg_number)
                if neg_number == pos_num - neg_num:
                    flag = False
                    break
        if not flag:
            break

    return srcs, dsts


edge_srcs = []
edge_dsts = []

# df1 = pd.read_csv('data/graph.csv')
df1 = pd.read_csv('graph_replace.csv')
pos_num, neg_num, srcs1, dsts1 = calculate_pos_sample_number(df1)

edge_src = torch.from_numpy(np.array(df1.ID_A.tolist()))
edge_dst = torch.from_numpy(np.array(df1.ID_B.tolist()))
# graph = dgl.heterograph({
#     ('drug', 'Level', 'drug'): (edge_src, edge_dst),
# })
graph = dgl.graph((edge_src, edge_dst))
print(graph)

srcs2, dsts2 = neg_sample(graph, pos_num, neg_num)
edge_srcs.append(srcs1 + srcs2)
edge_dsts.append(dsts1 + dsts2)

neg = pd.DataFrame()
neg['ID_A'] = edge_srcs[0]
neg['ID_B'] = edge_dsts[0]
neg['Level'] = ['Unknown'] * pos_num
neg.to_csv('1603neg_sample.csv', mode='w', index=False)
