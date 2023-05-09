import dgl
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

df5 = pd.read_csv('../data_binary_classify/1603SimilarityMatrix.csv', index_col=0)  # 不读索引列

# 数据归一化
scaler = MinMaxScaler()
# scaler = StandardScaler()
df_standard1 = scaler.fit_transform(df5)
drug_feature_similarity = SimilarityEncoder(df_standard1)
# node2特征
df6 = pd.read_csv('../data_binary_classify/1603embeddingResult.csv', index_col=0)  # 不读索引列
df_standard2 = scaler.fit_transform(df6.values)
drug_feature_node2vec = torch.tensor(df_standard2, dtype=torch.float)
# mpnn特征
featu = np.load('../data_binary_classify/mpnnPretain.npy')
df_standard3 = scaler.fit_transform(featu)
drug_feature_mpnn = torch.tensor(df_standard3, dtype=torch.float)

df1 = pd.read_csv('../data_binary_classify/graph_known_replace.csv')
known_edge_src_list = df1.ID_A.tolist()
known_edge_dst_list = df1.ID_B.tolist()
known_edge_src = torch.from_numpy(np.array(known_edge_src_list))
known_edge_dst = torch.from_numpy(np.array(known_edge_dst_list))

# df2 = pd.read_csv('../data_binary_classify/1603neg_sample.csv')
df2 = pd.read_csv('../data_binary_classify/graph_Unknown_replace.csv')
Unknown_edge_src_list = df2.ID_A.tolist()
Unknown_edge_dst_list = df2.ID_B.tolist()
Unknown_edge_src = torch.from_numpy(np.array(Unknown_edge_src_list))
Unknown_edge_dst = torch.from_numpy(np.array(Unknown_edge_dst_list))

graph_predict = dgl.heterograph({
    ('drug', 'Known', 'drug'): (known_edge_src, known_edge_dst),
    ('drug', 'Unknown', 'drug'): (Unknown_edge_src, Unknown_edge_dst)
})

num = graph_predict.num_nodes

graph_predict.ndata['drug_feature_similarity'] = drug_feature_similarity  # 添加drug_similarity特征向量
graph_predict.ndata['drug_feature_node2vec'] = drug_feature_node2vec  # 添加drug_node2vec特征向量
graph_predict.ndata['drug_feature_structureMpnn'] = drug_feature_mpnn  # 添加drug_mpnn特征向量


print("predict graph:\n", graph_predict)
