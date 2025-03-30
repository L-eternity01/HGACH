import logging
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import GCNConv
import timm
from transformers import BertModel, BertTokenizer
import clip  # 使用CLIP库
from clip import load

# HCHA 版本的 G 构建
def edge_list(G):
    list_e = []
    cla = []
    N = len(G[0])
    for i in range(N):
        for j in range(N):
            if G[i][j] != -1.5:
                list_e.append(i)
                cla.append(j % N)  # 强制限制超边索引在 [0, N-1] 内

    res = []
    res.append(list_e)
    res.append(cla)

    # # 添加调试信息
    # print("Max hyperedge index (cla):", max(cla) if cla else -1)
    # print("Number of nodes (N):", N)
    # assert max(cla) < N, f"Hyperedge index {max(cla)} exceeds node count {N}"
    return res

#KDD 22 版本的 G 构建
def get_hyperedge_attr(features, hyperedge_index, type='var'):
    # input: features: tensor N x F; hyperedge_index: 2 x |sum of all hyperedge size|
    # return hyperedge_attr: tensor, M x F
    #features = torch.FloatTensor([[0, 0.1, 0.2], [1.1, 1.2, 1.3], [2., 2.1, 2.2], [3.1,3.2,3.3], [4, 4.1, 4.2], [5,5,5]])
    #hyperedge_index = torch.LongTensor([[0,1,0,3,4,5,1],[0,0,1,1,1,2,2]])
    if type == 'var':
        # hyperedge_attr = features[hyperedge_index[0]]  # |sum of all hyperedge size| x F
        # index_start =  # M, the start index of every hyperedge
        # hyperedge_attr = torch.tensor_split(hyperedge_attr, index_start)  #
        hyperedge_attr = None
        samples = features[hyperedge_index[0]]
        labels = hyperedge_index[1]

        labels = labels.view(labels.size(0), 1).expand(-1, samples.size(1))
        # print(labels)
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        # # 添加调试信息
        # print("unique_labels.size(0):", unique_labels.size(0))
        # print("labels.max():", labels.max())
        # assert labels.max() < unique_labels.size(0), f"Index {labels.max()} out of bounds for size {unique_labels.size(0)}"


        # print(unique_labels, labels_count)
        hyperedge_attr = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, samples)
        # print(hyperedge_attr)
        hyperedge_attr = hyperedge_attr / labels_count.float().unsqueeze(1)
    return hyperedge_attr




class ImgNet_V2(nn.Module):
    def __init__(self, code_len):
        super(ImgNet_V2, self).__init__()
       # self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet = torchvision.models.vgg19(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-2])
        self.hgac = HypergraphConv(4096, 4096, use_attention=True, heads = 8, concat=False) 
        self.hgc1 = HypergraphConv(4096, code_len)
        # add batchnorm
        self.bn = nn.BatchNorm1d(4096)
        self.alpha = 1.0

    def forward(self, x, G=None):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        # (16, 4096)
        feat = self.alexnet.classifier(x)
        if G is None:
            return feat, 0, 0
        G = torch.LongTensor(edge_list(G)).cuda()
        if G.size(1) == 0:  # 检查超边索引是否为空
            print("Warning: Empty hyperedge_index, returning default values")
            return feat, feat, torch.zeros(x.size(0), self.hgc1.out_channels).cuda()
        try:
            hyperedge_attr = get_hyperedge_attr(feat, hyperedge_index=G, type='var')
            hid_with_att = self.hgac(feat, G, hyperedge_attr=hyperedge_attr)
        except Exception as e:
            logging.error(f"Error in forward: {str(e)}")
            raise  # 抛出异常以查看详细信息
            
        # 改用 BN
        hid_with_att = self.bn(hid_with_att)
        hid_with_att = F.leaky_relu(hid_with_att, negative_slope=0.2)
        hid_with_att = F.dropout(hid_with_att, 0.6, training=self.training)

        # hid_cat = torch.cat([feat, hid_with_att], dim=1)
        hid_cat = (feat + hid_with_att) * 0.5
        # no relu
        hid = self.hgc1(hid_cat, G)
        code = F.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet_V2(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet_V2, self).__init__()
        self.hgc1 = HypergraphConv(txt_feat_len, 4096)
        self.hgac = HypergraphConv(4096, 4096, use_attention=True, heads=8, concat=False)
        self.hgc2 = HypergraphConv(4096, code_len)
        self.alpha = 1.0
    def forward(self, x, G):
        G = torch.LongTensor(edge_list(G)).cuda()
        feat = F.relu(self.hgc1(x, G))
        hyperedge_attr = get_hyperedge_attr(feat, hyperedge_index=G, type='var')
        hid_with_att = self.hgac(feat, G, hyperedge_attr=hyperedge_attr)
        hid_with_att = F.dropout(hid_with_att, 0.6, training=self.training)
        hid_cat = feat + hid_with_att
        hid = self.hgc2(hid_cat, G)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code
    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

# # HCHA 版本的 model without attention
class ImgNet_V1(nn.Module):
    def __init__(self, code_len):
        super(ImgNet_V1, self).__init__()
        # self.alexnet = torchvision.models.alexnet(pretrained=True)
        #self.alexnet = torchvision.models.vgg16(pretrained=True)
        self.alexnet = torchvision.models.vgg19(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-2])
        self.hgc1 = HypergraphConv(4096,code_len)
        self.alpha = 1.0 
        

    def forward(self, x, G = None):        
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        # feat = x
        if G is None:
            return feat, 0, 0
        G = torch.LongTensor(edge_list(G)).cuda()
        hid = self.hgc1(feat, G)
        code = F.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet_V1(nn.Module):

    def __init__(self, code_len, txt_feat_len):
        super(TxtNet_V1, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.hgc1 = HypergraphConv(txt_feat_len, 4096)
        self.hgc2 = HypergraphConv(4096, code_len)
        self.alpha = 1.0


    def forward(self, x, G):
        G = torch.LongTensor(edge_list(G)).cuda()
        feat = F.relu(self.hgc1(x, G))
        hid = self.hgc2(feat, G)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)



