import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import datasets
import settings
from model import  ImgNet_V1, TxtNet_V1,ImgNet_V2,TxtNet_V2
from metric import compress_wiki, compress_nus,compress, calculate_top_map, load_feature_construct_H, generate_G_from_H,calculate_map
import os.path as osp
import random
import numpy as np
import copy
from tools import build_G_from_S, generate_robust_S



#返回 归一化后 的数据集图文特征，其中图像特征为基于预训练模型提取的特征，文本特征为数据集自带的特征
def extract_features(img_model, dataloader, feature_loader):
    """
    Extract features.
    """
    if settings.DATASET == "WIKI":
        sample_num = len(feature_loader.dataset.label)
    else:
        sample_num = feature_loader.dataset.train_labels.shape[0]
    img_model.cuda().eval()
    img_features = torch.zeros(sample_num, 4096).cuda()
    if settings.DATASET == "MIRFlickr":
        txt_features = torch.zeros(sample_num, 1386).cuda()
    if settings.DATASET == "WIKI":
        txt_features = torch.zeros(sample_num, 10).cuda()
    if settings.DATASET == "NUSWIDE":
        txt_features = torch.zeros(sample_num, 1000).cuda()
    if settings.DATASET == "MSCOCO":
        txt_features = torch.zeros(sample_num, 2000).cuda()
    with torch.no_grad():
        for i, (img, F_T, _, index) in enumerate(dataloader):
            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())
            img_features[index, :], _, _ = img_model(img)
            txt_features[index, :] = F_T

    return F.normalize(img_features), F.normalize(txt_features)


#对联合模态语义相似度矩阵进行随机游走，此处代码参考“https://github.com/rongchengtu1/MLS3RDUH”
def random_walk(sim1, dim):
    sim1 = sim1.cpu()
    final_sim = sim1 * 1.0
    #sim 为去掉主对角线上所有1后的相似度矩阵
    sim = sim1 - torch.eye(dim).type(torch.FloatTensor)
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        #top20之后的结果
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        sim[i, :] = top[0, :] * zero
    #矩阵的 > 判断符后返回的结果是   元素为1 或者 0 的矩阵
    A = (sim > 0.0001).type(torch.FloatTensor)
    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)
    aa = dim - (sum_row > 0).sum()
    kk = sum_row.sort()[1]
    res_ind = list(range(dim))
    for ind in range(aa):
        res_ind.remove(kk[ind])
    res_ind = random.sample(res_ind, dim - aa)
    ind_to_new_id = {}
    for i in range(dim - aa):
        ind_to_new_id[i] = res_ind[i]
    res_ind = (torch.from_numpy(np.asarray(res_ind))).type(torch.LongTensor)
    sim = sim[res_ind, :]
    sim = sim[:, res_ind]
    sim20 = {}
    dim = dim - aa
    top = torch.rand((1, dim)).type(torch.FloatTensor)
    for i in range(dim):
        top[0, :] = sim[i, :]
        top20 = top.sort()[1][0]
        zero = torch.zeros(dim).type(torch.FloatTensor)
        zero[top20[-nnk:]] = 1.0
        k = list(top20[-nnk:])
        sim20[i] = k
        sim[i, :] = top[0, :] * zero
    A = (sim > 0.0001).type(torch.FloatTensor)

    A = A * (A.t())
    A = A * sim
    sum_row = A.sum(1)

    sum_row = sum_row.pow(-0.5)
    sim = torch.diag(sum_row)
    A = A.mm(sim)
    A = sim.mm(A)
    alpha = 0.99
    manifold_sim = (1 - alpha) * torch.inverse(torch.eye(dim).type(torch.FloatTensor) - alpha * A)

    manifold20 = {}
    for i in range(dim):
        top[0, :] = manifold_sim[i, :]
        top20 = top.sort()[1][0]
        k = list(top20[-nno:])
        manifold20[i] = k
    for i in range(len(sim20)):
        if (i+1) % 500 == 0 :
            settings.logger.info('Epoch [%d/%d]'% (i+1, len(sim20)))
        aa = len(manifold20[i])
        zz = copy.deepcopy(manifold20[i])
        ddd = []
        for k in range(aa):
            if zz[k] in sim20[i]:
                sim20[i].remove(zz[k])
                manifold20[i].remove(zz[k])
                ddd.append(ind_to_new_id[int(zz[k])])
        j = ind_to_new_id[i]
        
        for l in ddd:
            final_sim[j, int(l)] = 1.0
            
        for l in sim20[i]:
            final_sim[j, ind_to_new_id[int(l)]] = 0.0


    # final_sim = ((final_sim + final_sim.t()) > 0.1).type(torch.FloatTensor) - ((final_sim + final_sim.t()) < -0.1).type(torch.FloatTensor)
    f1 = (final_sim > 0.999).type(torch.FloatTensor)
    f1 = ((f1 + f1.t()) > 0.999).type(torch.FloatTensor)
    f2 = (final_sim < 0.0001).type(torch.FloatTensor)
    f2 = ((f2 + f2.t()) > 0.999).type(torch.FloatTensor)
    final_sim = final_sim * (1. - f2)
    final_sim = final_sim * (1. - f1) + f1
    
    return final_sim


if settings.DATASET == "WIKI":
    train_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_train_transform)
    test_dataset = datasets.WIKI(root=settings.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
    database_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)
    feature_dataset = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)
    feature_dataset_test = datasets.WIKI(root=settings.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
    feature_dataset_database = datasets.WIKI(root=settings.DATA_DIR, train=True, transform=datasets.wiki_test_transform)
if settings.DATASET == "MIRFlickr":
    # 加载有标签训练集 (30%)
    train_dataset_labeled = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform, labeled=True)  # 30%有标签数据
    print(f"有标签数据数量: {len(train_dataset_labeled)}")
    
    # 加载无标签训练集 
    train_dataset_unlabeled = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform, labeled=False)  
    print(f"无标签数据数量: {len(train_dataset_unlabeled)}")
    # 加载测试集和数据库集（这些数据集不受标签划分的影响）
    test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
    database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

    # 特征提取相关的数据集（通常用于测试阶段，依赖于图像转换）
    feature_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_test_transform)
    feature_dataset_test = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
    feature_dataset_database = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)


if settings.DATASET == "NUSWIDE":
    # 加载有标签训练集 (30%)
    train_dataset_labeled = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform, labeled=True)  # 30%有标签数据
    #print(f"有标签数据数量: {len(train_dataset_labeled)}")
    
    # 加载无标签训练集
    train_dataset_unlabeled = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform, labeled=False)
    #print(f"无标签数据数量: {len(train_dataset_unlabeled)}")
    
    # 加载测试集和数据库集（这些数据集不受标签划分的影响）
    test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
    database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)
    
    # 特征提取相关的数据集（通常用于测试阶段，依赖于图像转换）
    feature_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_test_transform)
    feature_dataset_test = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
    feature_dataset_database = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)
if settings.DATASET == "MSCOCO":
    # 加载有标签训练集 (30%)
    train_dataset_labeled = datasets.MSCOCO(train=True, transform=datasets.coco_train_transform, labeled=True)  # 30%有标签数据
    #print(f"有标签数据数量: {len(train_dataset_labeled)}")
    
    # 加载无标签训练集
    train_dataset_unlabeled = datasets.MSCOCO(train=True, transform=datasets.coco_train_transform, labeled=False)
    #print(f"无标签数据数量: {len(train_dataset_unlabeled)}")
    
    # 加载测试集和数据库集（这些数据集不受标签划分的影响）
    test_dataset = datasets.MSCOCO(train=False, database=False, transform=datasets.coco_test_transform)
    database_dataset = datasets.MSCOCO(train=False, database=True, transform=datasets.coco_test_transform)
    
    # 特征提取相关的数据集（通常用于测试阶段，依赖于图像转换）
    feature_dataset = datasets.MSCOCO(train=True, transform=datasets.coco_test_transform)
    feature_dataset_test = datasets.MSCOCO(train=False, database=False, transform=datasets.coco_test_transform)
    feature_dataset_database = datasets.MSCOCO(train=False, database=True, transform=datasets.coco_test_transform)


# 有标签数据的训练加载器
train_loader_labeled = torch.utils.data.DataLoader(dataset=train_dataset_labeled,
                                                   batch_size=settings.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=settings.NUM_WORKERS,
                                                   drop_last=True)

# 无标签数据的训练加载器
train_loader_unlabeled = torch.utils.data.DataLoader(dataset=train_dataset_unlabeled,
                                                     batch_size=settings.BATCH_SIZE,
                                                     shuffle=True,
                                                     num_workers=settings.NUM_WORKERS,
                                                     drop_last=True)

# 测试集数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=settings.NUM_WORKERS)

# 数据库集数据加载器
database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=settings.BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=settings.NUM_WORKERS)

# 特征提取相关的数据加载器（测试阶段）
feature_loader = torch.utils.data.DataLoader(dataset=feature_dataset,
                                             batch_size=settings.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=settings.NUM_WORKERS)

feature_loader_test = torch.utils.data.DataLoader(dataset=feature_dataset_test,
                                                 batch_size=settings.BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=settings.NUM_WORKERS)

feature_loader_database = torch.utils.data.DataLoader(dataset=feature_dataset_database,
                                                     batch_size=settings.BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=settings.NUM_WORKERS)




# train dataset's random walk
FeatNet_I = ImgNet_V1(code_len=settings.CODE_LEN)
if settings.DATASET == "WIKI":
    sample_num = len(feature_loader.dataset.label)
else:
    sample_num = feature_loader.dataset.train_labels.shape[0]

global_imgs, global_txts = extract_features(FeatNet_I, feature_loader, feature_loader)

nnk = int(sample_num * settings.nnk)
nno = int(sample_num * settings.nnk * 1.5) #1630 sample_num=13583


feature_img = global_imgs
feature_txt = global_txts
dim = sample_num

F_I = feature_img
S_I = F_I.mm(F_I.t())
F_T = feature_txt
S_T = F_T.mm(F_T.t())

settings.logger.info('dataset %s, nnk %.4f, nno %.4f, total epoch %d, eval interval %d, sim1 %.4f' % (settings.DATASET, settings.nnk, nno,  settings.NUM_EPOCH, settings.EVAL_INTERVAL,settings.sim1))


S_high_crs = F.normalize(S_I).mm(F.normalize(S_T).t())
if settings.DATASET == "MIRFlickr" or settings.DATASET == "MSCOCO":
    #sim1 = 0.5 * S_I + 0.1 * S_T + 0.4 * (S_high_crs + S_high_crs.t()) / 2
    sim1 = settings.A * S_I + settings.B * S_T + settings.C * (S_high_crs + S_high_crs.t()) / 2
    sim1 = sim1  * 1.4
# else settings.DATASET == "NUSWIDE":
else:
    #sim1 = 0.4 * S_I + 0.3 * S_T + 0.3 * (S_high_crs + S_high_crs.t()) / 2
    sim1 = settings.A * S_I + settings.B * S_T + settings.C * (S_high_crs + S_high_crs.t()) / 2
    sim1 = sim1  * 1.4
# else:
#     sim1 = 0.4 * S_I + 0.3 * S_T + 0.3 * (S_high_crs + S_high_crs.t()) / 2
#     sim = sim1 * 1.4
#final_sim = random_walk(sim1, dim)  #随机游走
final_sim = sim1

if settings.DATASET == "WIKI":
    sample_num = len(feature_loader_test.dataset.label)
else:
    sample_num = feature_loader_test.dataset.train_labels.shape[0]

global_imgs_test, global_txts_test = extract_features(FeatNet_I, feature_loader_test, feature_loader_test)


feature_img_test = global_imgs_test
feature_txt_test = global_txts_test

F_I_test = feature_img_test
S_I_test = F_I_test.mm(F_I_test.t())
F_T_test = feature_txt_test
S_T_test = F_T_test.mm(F_T_test.t())

S_high_crs_test = F.normalize(S_I_test).mm(F.normalize(S_T_test).t())
if settings.DATASET == "MIRFlickr" or settings.DATASET == "MSCOCO":

    #sim1_test = 0.5 * S_I_test + 0.1 * S_T_test + 0.4 * (S_high_crs_test + S_high_crs_test.t()) / 2
    sim1_test = settings.A * S_I_test + settings.B* S_T_test + settings.C * (S_high_crs_test + S_high_crs_test.t()) / 2
    sim1_test = sim1_test  * 1.4
else:
    #sim1_test = 0.4 * S_I_test + 0.3 * S_T_test + 0.3 * (S_high_crs_test + S_high_crs_test.t()) / 2
    sim1_test = settings.A * S_I_test + settings.B * S_T_test + settings.C * (S_high_crs_test + S_high_crs_test.t()) / 2
    sim1_test = sim1_test  * 1.4
# else:
#     sim1 = 0.4 * S_I_test + 0.3 * S_T_test + 0.3 * (S_high_crs_test + S_high_crs_test.t()) / 2
#     sim1_test = sim1  * 1.4
# final_sim_test = 2 * sim1_test -1
final_sim_test = sim1_test


#################### 数据库样本的G构建 ##############

# NUSWIDE 时不使用全局数据库样本
if settings.DATASET != 'NUSWIDE' and settings.DATASET != 'MSCOCO':
        if settings.DATASET == "WIKI":
            sample_num = len(feature_loader_test.dataset.label)
        else:
            sample_num = feature_loader_database.dataset.train_labels.shape[0]

        global_imgs_database, global_txts_database = extract_features(FeatNet_I, feature_loader_database, feature_loader_database)


        feature_img_database = global_imgs_database
        feature_txt_database = global_txts_database

        F_I_database = feature_img_database
        S_I_database = F_I_database.mm(F_I_database.t())
        F_T_database = feature_txt_database
        S_T_database = F_T_database.mm(F_T_database.t())

        S_high_crs_database = F.normalize(S_I_database).mm(F.normalize(S_T_database).t())
        if settings.DATASET == "MIRFlickr":
            #sim1_database = 0.5 * S_I_database + 0.1 * S_T_database + 0.4 * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = settings.A * S_I_database + settings.B * S_T_database + settings.C * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = sim1_database  * 1.4
        elif settings.DATASET == "NUSWIDE":
            #sim1_database = 0.4 * S_I_database + 0.3 * S_T_database + 0.3 * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = settings.A * S_I_database + settings.B * S_T_database + settings.C * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = sim1_database  * 1.4
        else:
            sim1 = 0.5 * S_I + 0.1 * S_T + 0.4 * (S_high_crs_database + S_high_crs_database.t()) / 2
            sim1_database = sim1 *  1.4
        final_sim_database = sim1_database
else:
        final_sim_database = 0


###########################################添加MITHloss
def create_similarity_matrix(labels):
    """
    根据标签生成标签相似度矩阵 S_l。
    :param labels: [batch_size, num_classes] 的标签矩阵，表示每个样本的多标签信息
    :return: 标签相似度矩阵 S_l， [batch_size, batch_size] 大小
    """
    # 计算标签矩阵的内积，如果有至少一个共同标签，则 S_l[i,j]=1，否则为 0
    S_l = (labels.mm(labels.t()) > 0).float()
    return S_l
#################################################MITH
def intra_channel_similarity_loss(B, S_l):
    """
    计算通道内相似性保持损失
    参数:
        B (torch.Tensor): 特征矩阵 (N, D)，可以是图像或文本特征
        S_l (torch.Tensor): 标签相似度矩阵 (N, N)
    返回:
        loss_intra (torch.Tensor): 通道内相似性损失
    """
    # 计算细粒度语义概念表示的内积 Ω_intra
    Ω_intra = 0.5 * torch.sum(B.unsqueeze(1) * B.unsqueeze(0), dim=2)
    
    # 计算通道内相似性保持损失
    loss_intra = -torch.mean(S_l * Ω_intra - torch.log(1 + torch.exp(Ω_intra)))
    return loss_intra


def inter_channel_similarity_loss(B_I, B_T, S_l):
    """
    计算通道间相似性保持损失
    参数:
        B_I (torch.Tensor): 图像特征矩阵 (N, D)
        B_T (torch.Tensor): 文本特征矩阵 (N, D)
        S_l (torch.Tensor): 标签相似度矩阵 (N, N)
    返回:
        loss_inter (torch.Tensor): 通道间相似性损失
    """
    # 计算粗粒度语义概念表示的内积 Θ_inter 和 Φ_inter
    Θ_inter = 0.5 * torch.sum(B_I.unsqueeze(1) * B_T.unsqueeze(0), dim=2)
    Φ_inter = 0.5 * torch.sum(B_T.unsqueeze(1) * B_I.unsqueeze(0), dim=2)
    
    # 计算通道间相似性保持损失
    loss_inter_I = -torch.mean(S_l * Θ_inter - torch.log(1 + torch.exp(Θ_inter)))
    loss_inter_T = -torch.mean(S_l * Φ_inter - torch.log(1 + torch.exp(Φ_inter)))
    loss_inter = (loss_inter_I + loss_inter_T) / 2
    return loss_inter





class Session:
    def __init__(self):
        self.logger = settings.logger
        torch.cuda.set_device(settings.GPU_ID)
        
        
        self.CodeNet_I = ImgNet_V1(code_len=settings.CODE_LEN)
        self.FeatNet_I = ImgNet_V1(code_len=settings.CODE_LEN)

        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_T = TxtNet_V1(code_len=settings.CODE_LEN, txt_feat_len=txt_feat_len)

        if settings.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE" or settings.DATASET == "MSCOCO":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=settings.LR_IMG, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=settings.LR_TXT, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
        
        if settings.DATASET == "MIRFlickr":
            final_sim1 = generate_robust_S(final_sim,settings.alpha,settings.beta)  
            self.final_sim = 2 * final_sim1 - 1.0
        else:
            self.final_sim = generate_robust_S(final_sim,settings.alpha,settings.beta) 




        

# # 画图
    def show_disturbtion(self):
        S = self.final_sim
        if isinstance(S, np.ndarray):
            S = torch.tensor(S, dtype=torch.float32).cuda()  # 如果是 NumPy 数组，先转换为 PyTorch 张量
        elif not S.is_cuda:
            S = S.cuda()  # 如果已经是 PyTorch 张量，且不在 GPU 上，则移动到 GPU

        # 计算 S_1 并用于展示
        S_1 = S.clone().detach()  # 使用 clone() 创建副本，避免修改原始张量
        S_1 = S_1.cuda()  # 确保 S_1 在 GPU 上

        # 去除对角线（使用 PyTorch 操作，不需要转为 NumPy）
        S_1 = S_1 - torch.diag(torch.diag(S_1))  # 使用 torch.diag 代替 np.diag
        S_1 = S_1.reshape(-1)

        # 计算 S_1 的均值和方差
        mean = S_1.mean().item()  # 使用 PyTorch 的函数计算均值
        std = S_1.std().item()

        # 使用 settings.threshold
        threshold = settings.threshold  # 确保 settings 已定义

        print("mean: ", mean)
        print("std: ", std)
        print("threshold: ", threshold)


        # # 在这里，使用settings.alpha和settings.beta代替硬编码的2
        # alpha = settings.alpha  # 从settings中获取alpha
        # beta = settings.beta    # 从settings中获取beta

        # # 调用generate_robust_S时传入alpha和beta
        # generate_robust_S(S, alpha, beta)


        # 打印有标签和无标签数据的数量
        print(f"有标签数据的数量: {len(train_loader_labeled.dataset)}")
        print(f"无标签数据的数量: {len(train_loader_unlabeled.dataset)}")

    def train(self, epoch):
        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (epoch + 1, settings.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        # 初始化总损失
        total_supervised_loss = 0.0
        total_unsupervised_loss = 0.0
        total_loss = 0.0

        # 处理有标签数据 
        for idx, (img, txt, labels, batch_ind) in enumerate(train_loader_labeled):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            labels = labels.cuda()
            S = self.final_sim[batch_ind, :]
            S = S[:, batch_ind]

            #S = S.cuda()

            # # 将 S_ 转换为 PyTorch 张量并移动到 GPU
            S = torch.tensor(S, dtype=torch.float32).cuda()

            # 生成标签相似度矩阵 S_l
            S_l = create_similarity_matrix(labels)

            # S = weighted_reconstruction(S, S_l, alpha=0.3)

            G = build_G_from_S(S, settings.K)

            batch_size = img.size(0)

            _, hid_I, code_I = self.CodeNet_I(img, G)
            _, hid_T, code_T = self.CodeNet_T(txt, G)

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            # 有监督损失计算
            intra_loss = settings.D * intra_channel_similarity_loss(B_I, S_l) + settings.E * intra_channel_similarity_loss(B_T, S_l)
            inter_loss = inter_channel_similarity_loss(B_I, B_T, S_l)
            loss_supervised = settings.INTRA * intra_loss + settings.INTER * inter_loss

            # 累加有监督损失
            total_supervised_loss += loss_supervised.item()

            # 反向传播
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            loss_supervised.backward()
            self.opt_I.step()
            self.opt_T.step()

        # 处理无标签数据
        for idx, (img, txt, _, batch_ind) in enumerate(train_loader_unlabeled):
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            S = self.final_sim[batch_ind, :]
            S = S[:, batch_ind]
            #S = S.cuda()

            # # 将 S_ 转换为 PyTorch 张量并移动到 GPU
            S = torch.tensor(S, dtype=torch.float32).cuda()

            G = build_G_from_S(S, settings.K)

            batch_size = img.size(0)

            _, hid_I, code_I = self.CodeNet_I(img, G)
            _, hid_T, code_T = self.CodeNet_T(txt, G)

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            # 无监督损失计算
            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())
            BT_BI = B_T.mm(B_I.t())

            loss1 = F.mse_loss(BI_BI, S)
            loss2 = (F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S)) * 0.5
            loss3 = F.mse_loss(BT_BT, S)
            diagonal = BI_BT.diagonal()
            all_1 = torch.rand((BT_BT.size(0))).fill_(1).cuda()
            loss4 = F.mse_loss(diagonal, 1.5 * all_1)

            # 总无监督损失
            loss_unsupervised = settings.LAMBDA1 * loss1 + settings.LAMBDA3 * loss2 + settings.LAMBDA2 * loss3 + settings.l4 * loss4

            # 累加无监督损失
            total_unsupervised_loss += loss_unsupervised.item()

            # 反向传播
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            loss_unsupervised.backward()
            self.opt_I.step()
            self.opt_T.step()

        total_loss = total_supervised_loss + total_unsupervised_loss

        # # #####消融：去掉有标签，为无监督
        #total_loss = 0 * total_supervised_loss + total_unsupervised_loss

        # 每个 epoch 的日志记录
        self.logger.info(f'Epoch [{epoch + 1}/{settings.NUM_EPOCH}], '
                        f'Supervised Loss: {total_supervised_loss:.4f}, '
                        f'Unsupervised Loss: {total_unsupervised_loss:.4f}, '
                        f'Total Loss: {total_loss:.4f}')

    def eval(self, avgScore):
            self.logger.info('--------------------Evaluation: Calculate top MAP -------------------')

            # # 将模型切换到评估模式，并移动到 GPU
            self.CodeNet_I.eval().cuda()  
            self.CodeNet_T.eval().cuda()
            # self.CodeNet_I.eval().cpu()
            # self.CodeNet_T.eval().cpu()


            # 根据不同的数据集选择压缩方法
            if settings.DATASET in ["MIRFlickr", "WIKI"]:
                re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(
                    database_loader, test_loader, self.CodeNet_I, self.CodeNet_T, 
                    database_dataset, test_dataset, final_sim_database, final_sim_test
                )
            elif settings.DATASET in ["NUSWIDE", "MSCOCO"]:
                re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_nus(
                    database_loader, test_loader, self.CodeNet_I, self.CodeNet_T, 
                    database_dataset, test_dataset, final_sim_test
                )
            else:
                self.logger.error(f"Unsupported dataset: {settings.DATASET}")
                return
            
            # # 添加调试信息
            # print("re_BI type:", type(re_BI), "shape:", re_BI.shape)
            # print("re_BT type:", type(re_BT), "shape:", re_BT.shape)
            # print("re_L type:", type(re_L), "shape:", re_L.shape)
            # print("qu_BI type:", type(qu_BI), "shape:", qu_BI.shape)
            # print("qu_BT type:", type(qu_BT), "shape:", qu_BT.shape)
            # print("qu_L type:", type(qu_L), "shape:", qu_L.shape)

            # 计算 MAP
            MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)

            # print("MAP_I2T type:", type(MAP_I2T))
            # print("MAP_T2I type:", type(MAP_T2I))

            avgScore[0] = (avgScore[0] * avgScore[2] + MAP_I2T) / (avgScore[2] + 1)
            avgScore[1] = (avgScore[1] * avgScore[2] + MAP_T2I) / (avgScore[2] + 1)
            avgScore[2] += 1
            if MAP_I2T + MAP_T2I >= avgScore[3] + avgScore[4]:
                avgScore[3] = MAP_I2T
                avgScore[4] = MAP_T2I

            self.logger.info(
                'MAP of Image to Text: %.3f, MAP of Text to Image: %.3f   '
                'avgI2T: %.4f avgT2I: %.4f bestPair:(%.3f, %.3f) evalNum:%d' % (
                    MAP_I2T, MAP_T2I, avgScore[0], avgScore[1],
                    avgScore[3], avgScore[4], avgScore[2]
                )
            )


    def save_checkpoints(self, step, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    
    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])


def main():
    random_id = 30
    torch.manual_seed(random_id)
    torch.cuda.manual_seed_all(random_id)
    settings.logger.info('random seed id: %d' % random_id)
    settings.logger.info('%.4f loss1, 1 loss2, %.4f loss3, %.4f loss4, %d bit, map@all!!!' % 
                        (settings.LAMBDA1, settings.LAMBDA2, settings.l4, settings.CODE_LEN))
    sess = Session()

    avgScore = [0.0, 0.0, 0, 0, 0]

    if settings.EVAL:
        sess.load_checkpoints()
        sess.eval(avgScore)
    else:
        for epoch in range(settings.NUM_EPOCH):
            sess.train(epoch)
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval(avgScore)

    # 保存最终结果
    with open('./HGACH/result/f25k.txt', 'a') as f:
        f.write('%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n' % 
                (settings.DATASET, settings.A, settings.B, settings.C, settings.threshold, settings.K,
                 settings.alpha, settings.beta, settings.D, settings.E, avgScore[3], avgScore[4], avgScore[3]+avgScore[4]))


def show():
        torch.manual_seed(30)
        torch.cuda.manual_seed_all(30)
        settings.logger.info('random seed id: %d'%30)
        settings.logger.info('%.4f loss1, 1 loss2, %.4f loss3, %.4f loss4, %d bit, map@all!!!' % (settings.LAMBDA1, settings.LAMBDA2, settings.l4, settings.CODE_LEN))
        sess = Session()
        #sess.show_disturbtion()
if __name__ == '__main__':
    # # 存储所有实验的结果
    # for i in np.arange(0.5, 1, 0.1):
    #     settings.A = i
    #     for j in np.arange(0.5, 1 - i, 0.1):
    #         settings.B = j
    #         settings.C = 1 - i - j  # 直接计算C，确保C是合理的
    # for i in np.arange(-0.5,0,0.1):
    #     settings.threshold = i
        
        # for j in np.arange(6,10,1):
        #     settings.K = j
        main()  # 调用main进行训练
    # for i in np.arange(1,4.5,0.5):
    #     settings.alpha = i
    #     for j in np.arange(2,4.5,0.5):
    #         settings.beta = j
            # main()  # 调用main进行训练
