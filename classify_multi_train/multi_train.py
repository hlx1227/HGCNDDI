import time
import random
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
import torch
import dgl
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
from multi_model import Model, HeteroMLPPredictor


def Train(random_seed, epochs, lr, eps, weight_decay):
    # CPU or GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU')
    else:
        device = torch.device('cpu')
        print('The code uses CPU')

    multi_class = 4
    # g = graph

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    Major_train_index = []
    Major_test_index = []
    Minor_train_index = []
    Minor_test_index = []
    Moderate_train_index = []
    Moderate_test_index = []

    Major_edge_index = np.arange(0, graph_predict.num_edges('Major'))
    Minor_edge_index = np.arange(0, graph_predict.num_edges('Minor'))
    Moderate_edge_index = np.arange(0, graph_predict.num_edges('Moderate'))

    for train_idx, test_idx in kf.split(Major_edge_index):
        Major_train_index.append(train_idx)
        Major_test_index.append(test_idx)
    for train_idx, test_idx in kf.split(Minor_edge_index):
        Minor_train_index.append(train_idx)
        Minor_test_index.append(test_idx)
    for train_idx, test_idx in kf.split(Moderate_edge_index):
        Moderate_train_index.append(train_idx)
        Moderate_test_index.append(test_idx)

    auc_result = []
    acc_result = []
    pre_result_micro = []
    pre_result_macro = []
    recall_result_micro = []
    recall_result_macro = []
    f1_result_micro = []
    f1_result_macro = []

    fprs = []
    tprs = []

    for i in range(5):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)

        train_graph, test_graph = heterograph_edge_subgraph(Major_train_index[i], Major_test_index[i],
                                                            Minor_train_index[i], Minor_test_index[i],
                                                            Moderate_train_index[i], Moderate_test_index[i])

        model = Model(1565, 128, 300, 100, 4, graph_predict.etypes).to(device)
        # model = Model(128, 100, 4, graph_predict.etypes).to(device)

        # dec_graph会返回一个异构图，它具有drug 这种节点类型，以及把它们之间的所有边的类型进行合并后的单一边类型。
        dec_train_graph = train_graph['drug', :, 'drug']
        train_edge_label = dec_train_graph.edata[dgl.ETYPE].long()  # 转为long类型

        dec_test_graph = test_graph['drug', :, 'drug']
        test_edge_label = dec_test_graph.edata[dgl.ETYPE].long()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99)
        # optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9)
        print('Make prediction for :', 'test_subgraph')

        for epoch in range(epochs):
            train_drug_feats = train_graph.nodes['drug'].data['drug_feature_similarity']
            train_drug_feats_node2 = train_graph.nodes['drug'].data['drug_feature_node2vec']
            train_drug_feats_mpnn = train_graph.nodes['drug'].data['drug_feature_structureMpnn']
            train_node_features_similarity = {'drug': train_drug_feats, 'drug': train_drug_feats}
            train_node_features_node2 = {'drug': train_drug_feats_node2, 'drug': train_drug_feats_node2}
            train_node_features_mpnn = {'drug': train_drug_feats_mpnn, 'drug': train_drug_feats_mpnn}
            logits = model(train_graph, train_node_features_similarity, train_node_features_node2,
                           train_node_features_mpnn, dec_train_graph)
            # logits = model(train_graph, train_node_features_node2, dec_train_graph)
            # if epoch < epochs // 2:  # loss前半部分用交叉熵，后半部分用focal_loss
            #     loss = F.cross_entropy(logits, train_edge_label).to(device)
            # else:
            #     loss = focal_loss(logits, train_edge_label).to(device)

            # weight = torch.tensor([1, 1, 0.7])
            # loss = F.cross_entropy(logits, train_edge_label, weight).to(device)
            loss = F.cross_entropy(logits, train_edge_label).to(device)
            # loss = focal_loss(logits, train_edge_label).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            lr = optimizer.param_groups[0]['lr']
            print("epoch={}, lr={}".format(epoch, lr))

            # test
            model.eval()
            with torch.no_grad():
                test_drug_feats = test_graph.nodes['drug'].data['drug_feature_similarity']
                test_drug_feats_node2 = test_graph.nodes['drug'].data['drug_feature_node2vec']
                test_drug_feats_mpnn = test_graph.nodes['drug'].data['drug_feature_structureMpnn']
                test_node_features = {'drug': test_drug_feats, 'drug': test_drug_feats}
                test_node_features_node2 = {'drug': test_drug_feats_node2, 'drug': test_drug_feats_node2}
                test_node_features_mpnn = {'drug': test_drug_feats_mpnn, 'drug': test_drug_feats_mpnn}
                test_output = model(test_graph, test_node_features, test_node_features_node2, test_node_features_mpnn,
                                    dec_test_graph)
                # train_output = model(train_graph, train_node_features, dec_train_graph)

                # test_drug_feats_1 = test_graph.nodes['drug'].data['drug_feature_similarity']
                # test_drug_feats_2 = test_graph.nodes['drug'].data['drug_feature_node2vec']
                # test_node_features_1 = {'drug': test_drug_feats_1, 'drug': test_drug_feats_1}
                # test_node_features_2 = {'drug': test_drug_feats_2, 'drug': test_drug_feats_2}
                # output = model(test_graph, test_node_features_1, test_node_features_2, dec_test_graph)

                classification = F.softmax(test_output, 1).to('cpu').data.numpy()
                a = classification[:, 1]
                predicted_labels = list(map(lambda x: np.argmax(x), classification))
                predicted_scores = list(map(lambda x: x[1], classification))

                # score_test = output.detach().numpy()
                # predicted_labels = list(map(lambda x: np.argmax(x), score_test))

                # Binarize the output
                Y_valid = label_binarize(test_edge_label, classes=[i for i in range(multi_class)])
                Y_pred = label_binarize(predicted_labels, classes=[i for i in range(multi_class)])

                fpr = dict()
                tpr = dict()
                roc_auc = dict()  # 用作绘图
                for i in range(multi_class):
                    # a = Y_valid[:, i]
                    # b = Y_pred[:, i]
                    fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                accuracy_test = accuracy_score(test_edge_label, predicted_labels)

                roc_auc = roc_auc_score(test_edge_label, classification, average='macro', multi_class='ovr')
                precision_test_micro = precision_score(test_edge_label, predicted_labels, average='micro')
                precision_test_weighted = precision_score(test_edge_label, predicted_labels, average='weighted')
                precision_test_macro = precision_score(test_edge_label, predicted_labels, average='macro')
                recall_test_micro = recall_score(test_edge_label, predicted_labels, average='micro')
                recall_test_weighted = recall_score(test_edge_label, predicted_labels, average='weighted')
                recall_test_macro = recall_score(test_edge_label, predicted_labels, average='macro')
                f1_test_micro = f1_score(test_edge_label, predicted_labels, average='micro')
                f1_test_weighted = f1_score(test_edge_label, predicted_labels, average='weighted')
                f1_test_macro = f1_score(test_edge_label, predicted_labels, average='macro')
                mcc = matthews_corrcoef(test_edge_label, predicted_labels)

                # print("loss= ,accuracy= ", loss.item())f1_test_macro {:.10f}
                tqdm.write(
                    'Epoch{:3d}   loss {:.5f}  acc {:.10f} auc {:.10f} precision_test_micro {:.10f}  precision_test_macro {:.10f} precision_test_weight {:.10f}  recall_test_micro {:.10f} recall_test_macro {:.10f} recall_test_weight {:.10f} f1_test_micro {:.10f}  f1_test_macro {:.10f} f1_test_weight {:.10f} MCC {:.10f}'
                        .format(epoch, loss.item(), accuracy_test, roc_auc, precision_test_micro, precision_test_macro,
                                precision_test_weighted,
                                recall_test_micro, recall_test_macro, recall_test_weighted, f1_test_micro,
                                f1_test_macro, f1_test_weighted, mcc))

        # print('Fold:', i + 1, 'Test Acc: %.4f' % accuracy_test, 'Test Pre: %.4f' % precision_test,
        #       'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test AUC: %.4f' % test_auc)
        #
        auc_result.append(roc_auc)
        acc_result.append(accuracy_test)
        pre_result_micro.append(precision_test_micro)
        pre_result_macro.append(precision_test_macro)
        recall_result_micro.append(recall_test_micro)
        recall_result_macro.append(recall_test_macro)
        f1_result_micro.append(f1_test_micro)
        f1_result_macro.append(f1_test_macro)
        #
        # fprs.append(fpr)
        # tprs.append(tpr)

    print('Training Finished')
    print('----------------------------------------------------------------------------------------------------------')

    return acc_result, auc_result, pre_result_micro, pre_result_macro, recall_result_micro, recall_result_macro, f1_result_micro, f1_result_macro, mcc, fpr, tpr


def edges_with_feature_one(edges):
    # Whether an edge has feature 1
    return (edges.data['h'] == 1.).squeeze(1)


def heterograph_edge_subgraph(Major_train_index, Major_test_index,
                              Minor_train_index, Minor_test_index,
                              Moderate_train_index, Moderate_test_index, ):
    train_graph = dgl.edge_subgraph(graph_predict, {'Major': Major_train_index, 'Minor': Minor_train_index,
                                                    'Moderate': Moderate_train_index})
    test_graph = dgl.edge_subgraph(graph_predict, {'Major': Major_test_index, 'Minor': Minor_test_index,
                                                   'Moderate': Moderate_test_index})
    return train_graph, test_graph


def focal_loss(preds, labels):  # 解决多分类的数据不均衡问题
    gamma = 2
    # assert preds.dim() == 2 and labels.dim()==1
    labels = labels.view(-1, 1)  # [B * S, 1]
    preds = preds.view(-1, preds.size(-1))  # [B * S, C]

    preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
    preds_softmax = torch.exp(preds_logsoft)  # softmax

    preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
    preds_logsoft = preds_logsoft.gather(1, labels)

    loss = -torch.mul(torch.pow((1 - preds_softmax), gamma),
                      preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

    loss = loss.mean()

    return loss


warnings.filterwarnings(action='ignore')  # 忽略warning

acc_result, auc_result, pre_result_micro, pre_result_macro, recall_result_micro, \
recall_result_macro, f1_result_micro, f1_result_macro, mcc, fpr, tpr = Train(
    random_seed=3,
    # epochs=200,
    epochs=200,
    lr=1e-3,
    # lr=2e-3,
    eps=1e-8,
    weight_decay=0)
# weight_decay=1e-3)

print(' Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
      ' Auc mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
      'Precision_micro mean: %.4f, variance: %.4f \n' % (np.mean(pre_result_micro), np.std(pre_result_micro)),
      'Precision_macro mean: %.4f, variance: %.4f \n' % (np.mean(pre_result_macro), np.std(pre_result_macro)),
      'Recall_micro mean: %.4f, variance: %.4f \n' % (np.mean(recall_result_micro), np.std(recall_result_micro)),
      'Recall_macro mean: %.4f, variance: %.4f \n' % (np.mean(recall_result_macro), np.std(recall_result_macro)),
      'F1-score_micro mean: %.4f, variance: %.4f \n' % (np.mean(f1_result_micro), np.std(f1_result_micro)),
      'F1-score_macro mean: %.4f, variance: %.4f \n' % (np.mean(f1_result_macro), np.std(f1_result_macro)),
      'MCC mean: %.4f, variance: %.4f \n' % (np.mean(mcc), np.std(mcc)))
