import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from model import edge_list
from tools import build_G_from_S
import settings


def compress_ab(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(train_loader):
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, _, _) in enumerate(test_loader):
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target)
        
    with torch.no_grad():
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = modeli(var_data_I)
        if not isinstance(code_I, torch.Tensor):
            raise TypeError("Model output is not a Tensor")
        code_I = torch.sign(code_I)

        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I,_ = modeli(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target)
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T,_ = modelt(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]
    
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

# def compress_nus(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, G_test):
#     logging.info(f"Test loader length: {len(test_loader)}")
#     logging.info(f"G_test shape: {G_test.shape}, expected test size: {2100}")

#     re_BI = list([])
#     re_BT = list([])
#     re_L = list([])
#     for _, (data_I, data_T, target, batch_ind) in enumerate(train_loader):
#         data_I = Variable(data_I.cuda())
#         F_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         feature_I, _, _= model_I(data_I)
#         F_I = F.normalize(feature_I)
#         F_T = F.normalize(F_T)
#         S_I = torch.mm(F_I, F_I.t())
#         S_T = torch.mm(F_T, F_T.t())
#         S_high_crs =  F.normalize(S_I).mm(F.normalize(S_T).t())
#         sim1_database = 0.4 * S_I + 0.3 * S_T + 0.3 * (S_high_crs + S_high_crs.t()) / 2
#         S = sim1_database  * 1.4

#         G = build_G_from_S(S, settings.K)
#         with torch.no_grad():
#             var_data_I = Variable(data_I.cuda())
#             _,_,code_I = model_I(var_data_I, G)
#         code_I = torch.sign(code_I)
#         re_BI.extend(code_I.cpu().data.numpy())
        
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         _,_,code_T = model_T(var_data_T, G)
#         code_T = torch.sign(code_T)
#         re_BT.extend(code_T.cpu().data.numpy())

#     qu_BI = list([])
#     qu_BT = list([])
#     qu_L = list([])
#     for _, (data_I, data_T, _, batch_ind) in enumerate(test_loader):
#         _S = G_test[batch_ind, :]
#         S = _S[:, batch_ind]

#         # 添加调试信息
#         print("G_test shape:", G_test.shape)
#         print("batch_ind:", batch_ind)
#         print("_S shape:", _S.shape)
#         print("S shape:", S.shape)
#         print("Batch size:", data_I.size(0))
#         # 验证 S 的形状
#         assert S.shape[0] == S.shape[1] == data_I.size(0), "S shape does not match batch size!"
#         # H = load_feature_construct_H(S)
#         # G = generate_G_from_H(H)
#         # G = torch.tensor(G, dtype=torch.float).cuda()
#         G = build_G_from_S(S, settings.K)
#         G = build_G_from_S(S, settings.K)
#         hyperedge_index = edge_list(G)
#         print("hyperedge_index[0] max:", max(hyperedge_index[0]) if hyperedge_index[0] else -1)
#         print("hyperedge_index[1] max:", max(hyperedge_index[1]) if hyperedge_index[1] else -1)

#         # 验证 G 的内容
#         print("G shape:", G.shape if hasattr(G, 'shape') else len(G))
#         G = torch.LongTensor(hyperedge_index).cuda()
#         with torch.no_grad():
#             var_data_I = Variable(data_I.cuda())
#             _,_,code_I = model_I(var_data_I, G)
#         code_I = torch.sign(code_I)
#         qu_BI.extend(code_I.cpu().data.numpy())
        
#         var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
#         _,_,code_T = model_T(var_data_T, G)
#         code_T = torch.sign(code_T)
#         qu_BT.extend(code_T.cpu().data.numpy())

#     re_BI = np.array(re_BI)
#     re_BT = np.array(re_BT)
#     re_L = train_dataset.train_labels

#     qu_BI = np.array(qu_BI)
#     qu_BT = np.array(qu_BT)
#     qu_L = test_dataset.test_labels
#     return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L
def compress_nus(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, G_test):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, batch_ind) in enumerate(train_loader):
        data_I = Variable(data_I.cuda())
        F_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        feature_I, _, _= model_I(data_I)
        F_I = F.normalize(feature_I)
        F_T = F.normalize(F_T)
        S_I = torch.mm(F_I, F_I.t())
        S_T = torch.mm(F_T, F_T.t())
        S_high_crs =  F.normalize(S_I).mm(F.normalize(S_T).t())
        sim1_database = 0.4 * S_I + 0.3 * S_T + 0.3 * (S_high_crs + S_high_crs.t()) / 2
        S = sim1_database  * 1.4

        G = build_G_from_S(S, settings.K)
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = model_I(var_data_I, G)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = model_T(var_data_T, G)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, _, batch_ind) in enumerate(test_loader):
        _S = G_test[batch_ind, :]
        S = _S[:, batch_ind]
        # H = load_feature_construct_H(S)
        # G = generate_G_from_H(H)
        # G = torch.tensor(G, dtype=torch.float).cuda()
        G = build_G_from_S(S, settings.K)
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = model_I(var_data_I, G)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = model_T(var_data_T, G)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, G_train, G_test):
    
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    re_L_wiki = list([])
    for _, (data_I, data_T, target, batch_ind) in enumerate(train_loader):
        S_ = G_train[batch_ind, :]
        S = S_[:, batch_ind]
        # H = load_feature_construct_H(S)
        # G = generate_G_from_H(H)
        # G = torch.tensor(G, dtype=torch.float).cuda()
        G = build_G_from_S(S, settings.K)
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = model_I(var_data_I, G)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = model_T(var_data_T, G)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, batch_ind) in enumerate(test_loader):
        _S = G_test[batch_ind, :]
        S = _S[:, batch_ind]
        # H = load_feature_construct_H(S)
        # G = generate_G_from_H(H)
        # G = torch.tensor(G, dtype=torch.float).cuda()
        G = build_G_from_S(S, settings.K)
        with torch.no_grad():
            var_data_I = Variable(data_I.cuda())
            _,_,code_I = model_I(var_data_I, G)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _,_,code_T = model_T(var_data_T, G)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)

    re_L = train_dataset.train_labels
    qu_L = test_dataset.train_labels
    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex)) 
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

def load_feature_construct_H(feature, m_prob=1, K_neigs=[10], is_probH=True, split_diff_scale=False):
    H = None
    tmp = construct_H_with_KNN(feature, K_neigs=K_neigs, split_diff_scale=split_diff_scale, is_probH=is_probH, m_prob=m_prob)
    H = hyperedge_concat(H, tmp)

    return H


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x.cpu())
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
#     dis_mat = dis_mat.cpu()
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G
    
def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

