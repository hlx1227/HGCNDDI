import time
import random
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
import torch
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import roc_curve, confusion_matrix, matthews_corrcoef
from sklearn.metrics import cohen_kappa_score, auc, accuracy_score, roc_auc_score, precision_score, f1_score, \
    recall_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import warnings
from torch.optim import lr_scheduler
# from DglHeterograph import graph
from DglPredictHeterograph import graph_predict
# from DglPredictHeter_overAUndersample import graph_predict
from model_test_twofeaturesaddmpnn import Model, HeteroMLPPredictor


def build_graph(model):
    scaler = MinMaxScaler()
    # node2特征
    df2 = pd.read_csv('../data_binary_classify/predict/predict_embedding_asknown.csv', index_col=0)  # 不读索引列
    df_standard2 = scaler.fit_transform(df2.values)
    drug_feature_node2vec = torch.tensor(df_standard2, dtype=torch.float)
    # mpnn特征
    featu = np.load('../data_binary_classify/predict/predictsamplempnnPretain.npy')
    df_standard3 = scaler.fit_transform(featu)
    drug_feature_mpnn = torch.tensor(df_standard3, dtype=torch.float)

    # build graph

    df1 = pd.read_csv('../data_binary_classify/predict/graph_known_replace_addnewdrug.csv')
    known_edge_src_list = df1.ID_A.tolist()
    known_edge_dst_list = df1.ID_B.tolist()
    known_edge_src = torch.from_numpy(np.array(known_edge_src_list))
    known_edge_dst = torch.from_numpy(np.array(known_edge_dst_list))

    df2 = pd.read_csv('../data_binary_classify/predict/graph_Unknown_replace.csv')
    Unknown_edge_src_list = df2.ID_A.tolist()
    Unknown_edge_dst_list = df2.ID_B.tolist()
    Unknown_edge_src = torch.from_numpy(np.array(Unknown_edge_src_list))
    Unknown_edge_dst = torch.from_numpy(np.array(Unknown_edge_dst_list))

    test_graph = dgl.heterograph({
        ('drug', 'Unknown', 'drug'): (Unknown_edge_src, Unknown_edge_dst),
        ('drug', 'Known', 'drug'): (known_edge_src, known_edge_dst)
    })
    # test_graph = dgl.graph((edge_src, edge_dst))

    test_graph.ndata['drug_feature_node2vec'] = drug_feature_node2vec  # 添加drug_node2vec特征向量
    test_graph.ndata['drug_feature_structureMpnn'] = drug_feature_mpnn
    print('test_graph', test_graph)
    dec_test_graph = test_graph['drug', :, 'drug']
    test_edge_label = dec_test_graph.edata[dgl.ETYPE].long()

    test_drug_feats_node2 = test_graph.nodes['drug'].data['drug_feature_node2vec']
    test_drug_feats_mpnn = test_graph.nodes['drug'].data['drug_feature_structureMpnn']
    test_node_features_node2 = {'drug': test_drug_feats_node2, 'drug': test_drug_feats_node2}
    test_node_features_mpnn = {'drug': test_drug_feats_mpnn, 'drug': test_drug_feats_mpnn}

    test_output = model(test_graph, test_node_features_node2, test_node_features_mpnn,
                        dec_test_graph)
    return test_output


model = Model(128, 300, 100, 4, graph_predict.etypes).to('cpu')
# max acc: 0.9112232527980604
state_dict = torch.load('best_model.pt')
model.load_state_dict(state_dict)

output = build_graph(model)
classification = F.softmax(output, 1).to('cpu').data.numpy()
predicted_labels = list(map(lambda x: np.argmax(x), classification))

# 输出结果
df = pd.DataFrame(data=classification[0:1603, 0:],
                  columns=['0', '1'])
df['labels_pred'] = predicted_labels[0:1603]
df.to_csv('pred_result.csv')
