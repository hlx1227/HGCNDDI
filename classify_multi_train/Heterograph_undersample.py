import dgl
import random
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import numpy as np
import pandas as pd


def SimilarityEncoder(df):
    # df_transpose = pd.DataFrame(df.values.T)
    a = np.array(df)
    # tensor = torch.tensor(np.array(df_transpose, dtype=np.double))
    tensor = torch.tensor(df, dtype=torch.float)
    return tensor


# df1 = pd.read_csv('data/graph.csv')
# edge_src_list = df1.ID_A.tolist()
# edge_dst_list = df1.ID_B.tolist()
# edge_src = torch.from_numpy(np.array(edge_src_list))
# edge_dst = torch.from_numpy(np.array(edge_dst_list))

# df5 = pd.read_csv('data/1972SimilarityMatrix.csv', index_col=0)  # 不读索引列

df5 = pd.read_csv('../data_multi_classify/SimilarityMatrixKnown.csv', index_col=0)  # 不读索引列

# 数据归一化
scaler = MinMaxScaler()
# scaler = StandardScaler()
df_standard1 = scaler.fit_transform(df5)
drug_feature_similarity = SimilarityEncoder(df_standard1)
# node2特征
df6 = pd.read_csv('../data_multi_classify/knownEmbeddingResult.csv', index_col=0)  # 不读索引列
df_standard2 = scaler.fit_transform(df6.values)
drug_feature_node2vec = torch.tensor(df_standard2, dtype=torch.float)
# mpnn特征
featu = np.load('../data_multi_classify/1603knownmpnnPretain.npy')
df_standard3 = scaler.fit_transform(featu)
drug_feature_mpnn = torch.tensor(df_standard3, dtype=torch.float)

# df1 = pd.read_csv('../data_multi_classify/graph_Major_replace.csv')
df1 = pd.read_csv('../data_multi_classify_undersample/major_under_8629.csv')
Major_edge_src_list = df1.ID_A.tolist()
Major_edge_dst_list = df1.ID_B.tolist()
Major_edge_src = torch.from_numpy(np.array(Major_edge_src_list))
Major_edge_dst = torch.from_numpy(np.array(Major_edge_dst_list))

# df2 = pd.read_csv('data/graph_Minor.csv')
df2 = pd.read_csv('../data_multi_classify_undersample/graph_Minor_replace.csv')
Minor_edge_src_list = df2.ID_A.tolist()
Minor_edge_dst_list = df2.ID_B.tolist()
Minor_edge_src = torch.from_numpy(np.array(Minor_edge_src_list))
Minor_edge_dst = torch.from_numpy(np.array(Minor_edge_dst_list))

# df3 = pd.read_csv('../data_multi_classify_undersample/under_Moderate_25472.csv')
df3 = pd.read_csv('../data_multi_classify_undersample/moderate_under_8629.csv')
# df3 = pd.read_csv('../data_multi_classify/graph_Moderate_replace.csv')
# df3 = pd.read_csv('../data_multi_classify_undersample/moderate_under_30000.csv')
Moderate_edge_src_list = df3.ID_A.tolist()
Moderate_edge_dst_list = df3.ID_B.tolist()
Moderate_edge_src = torch.from_numpy(np.array(Moderate_edge_src_list))
Moderate_edge_dst = torch.from_numpy(np.array(Moderate_edge_dst_list))

graph_predict = dgl.heterograph({
    ('drug', 'Major', 'drug'): (Major_edge_src, Major_edge_dst),
    ('drug', 'Minor', 'drug'): (Minor_edge_src, Minor_edge_dst),
    ('drug', 'Moderate', 'drug'): (Moderate_edge_src, Moderate_edge_dst),
})

num = graph_predict.num_nodes

graph_predict.ndata['drug_feature_similarity'] = drug_feature_similarity  # 添加drug_similarity特征向量
graph_predict.ndata['drug_feature_node2vec'] = drug_feature_node2vec  # 添加drug_node2vec特征向量
graph_predict.ndata['drug_feature_structureMpnn'] = drug_feature_mpnn  # 添加drug_mpnn特征向量

# # 对moderate下采样
# edges_Moderate = graph_predict.edges(etype='Moderate', form='eid')  # 获取etype为’Unknown‘的所有边
# length_Moderate = graph_predict.num_edges('Moderate')
# # b = random.sample(list(range(0, length)), length // 2)
# sampled_Moderate = torch.tensor(random.sample(list(range(0, length_Moderate)), 40000), dtype=torch.int32)
# graph_predict.remove_edges(sampled_Moderate, etype='Moderate')

# # 对major下采样
# edges_Major = graph_predict.edges(etype='Major', form='eid')  # 获取etype为’Unknown‘的所有边
# length_Major = graph_predict.num_edges('Major')
# # b = random.sample(list(range(0, length)), length // 2)
# sampled_Major = torch.tensor(random.sample(list(range(0, length_Major)), 5472), dtype=torch.int32)
# graph_predict.remove_edges(sampled_Major, etype='Major')

# a = graph.nodes
# b = graph.dsttypes
# c = graph.ntypes
# d = graph.etypes
print("predict graph:\n", graph_predict)
