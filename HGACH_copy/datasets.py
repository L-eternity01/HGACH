import torch
import cv2
import scipy.io as scio
from PIL import Image
import settings
import numpy as np
import scipy.io as scio
from torchvision import transforms
import h5py
np.random.seed(42)

#############################无标签10000
if settings.DATASET == "MIRFlickr":
    # 设置随机种子
    np.random.seed(42)
    
    # 加载标签数据和文本数据
    label_set = scio.loadmat(settings.LABEL_DIR)
    label_set = np.array(label_set['LAll'], dtype=np.float32)
    txt_set = scio.loadmat(settings.TXT_DIR)
    txt_set = np.array(txt_set['YAll'], dtype=np.float32)

    first = True
    # 从24个不同的label中随机选取2000/10000个样本作为测试集和训练集
    for label in range(label_set.shape[1]):
        index = np.where(label_set[:, label] == 1)[0]
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:68]  # 前68个样本作为测试集
            train_index = index[68:68 + 340]  # 接下来的340个样本作为训练集
            first = False
        else:
            ind = np.array([i for i in list(index) if i not in (list(train_index)+list(test_index))])
            test_index = np.concatenate((test_index, ind[:84]))  # 为测试集添加84个样本  68 + 84*23=2000 
            train_index = np.concatenate((train_index, ind[84:84 + 420]))  # 为训练集添加420个样本 340+420*23=10000
        

    # 剩下的样本作为数据库
    database_index = np.array([i for i in range(label_set.shape[0]) if i not in test_index])
    
    # 如果训练集的样本数量不足10000，则从数据库中随机抽取差额数量的样本补足
    if train_index.shape[0] < 10000:
        pick = np.array([i for i in database_index if i not in train_index])
        N = pick.shape[0]
        perm = np.random.permutation(N)
        pick = pick[perm]
        res = 10000 - train_index.shape[0]
        train_index = np.concatenate((train_index, pick[:res]))

    # 计算有标签数据的数量（30%）
    train_size = len(train_index)
    num_labeled_samples = int(0.3 * train_size)
    labeled_indices = np.random.choice(train_index, num_labeled_samples, replace=False)
    unlabeled_indices = train_index  # 无标签数据为训练集的所有样本
    # ## 获取剩下的样本作为无标签样本
    # unlabeled_indices = np.setdiff1d(train_index, labeled_indices)
    print(f"训练集总数量: {len(train_index)}")

    # 数据转换
    mir_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mir_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    txt_feat_len = txt_set.shape[1]
    class MIRFlickr(torch.utils.data.Dataset):
        def __init__(self, transform=None, target_transform=None, train=True, database=False, labeled=False):
            self.transform = transform
            self.target_transform = target_transform
            self.labeled = labeled  # 标志，表示是否为有标签数据

            if train:
                self.train_labels = label_set[train_index]
                self.train_index = train_index
                self.txt = txt_set[train_index]

                if labeled:
                    # 选择 30% 有标签数据
                    self.train_index = labeled_indices
                    self.train_labels = label_set[labeled_indices]
                    self.txt = txt_set[labeled_indices]
                else:
                    # 无标签数据为全部训练集
                    self.train_index = unlabeled_indices
                    self.train_labels = label_set[unlabeled_indices]
                    self.txt = txt_set[unlabeled_indices]

            elif database:
                self.train_labels = label_set[database_index]
                self.train_index = database_index
                self.txt = txt_set[database_index]
            else:
                self.train_labels = label_set[test_index]
                self.train_index = test_index
                self.txt = txt_set[test_index]

        def __getitem__(self, index):
            mirflickr = h5py.File(settings.IMG_DIR, 'r', libver='latest', swmr=True)
            img, target = mirflickr['IAll'][self.train_index[index]], self.train_labels[index]
            img = Image.fromarray(np.transpose(img, (2, 1, 0)))
            mirflickr.close()

            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)
        

##########################################################:无标签10500
if settings.DATASET == "NUSWIDE":
        # 设置随机种子
    np.random.seed(42)

    label_set = scio.loadmat(settings.LABEL_DIR)
    label_set = np.array(label_set['LAll'], dtype=np.float32)

    # # 打印数据集的形状和样本数量
    # print("Label set shape:", label_set.shape)
    # print("Number of labels:", label_set.shape[0])  # 样本数量
    # print("Number of label categories:", label_set.shape[1])  # 类别数量
    
    # txt_file = h5py.File(settings.TXT_DIR)
    txt_file = scio.loadmat(settings.TXT_DIR)
    txt_set = np.array(txt_file['YAll']).transpose()
    print("txt_set shape before transpose:", txt_set.shape)


    txt_set = txt_set.T

    print("txt_set shape after transpose:", txt_set.shape)

    # 打印 txt_set 中包含的元素总数
    print("Number of elements in txt_set:", txt_set.size)

    #txt_file.close()


    first = True

    for label in range(label_set.shape[1]):
        index = np.where(label_set[:,label] == 1)[0]
        
        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]
        
        if first:
            test_index = index[:100]
            train_index = index[200:700]
            first = False
        else:
            ind = np.array([i for i in list(index) if i not in (list(train_index)+list(test_index))])
            test_index = np.concatenate((test_index, ind[:100]))
            train_index = np.concatenate((train_index, ind[200:700]))

        
    database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])

    indexTest = test_index
    indexDatabase = database_index
    indexTrain = train_index

    # 打印最终的训练集、测试集和数据库集的样本数
    print(f"Final test set size: {len(indexTest)}")
    print(f"Final train set size: {len(indexTrain)}")
    print(f"Final database set size: {len(indexDatabase)}")

    # 计算有标签数据的数量（30%）
    train_size = len(train_index)
    num_labeled_samples = int(0.3 * train_size)

    # 随机选择 30% 的样本作为有标签数据
    labeled_indices = np.random.choice(train_index, num_labeled_samples, replace=False)

    # 无标签数据为训练集的所有样本
    unlabeled_indices = train_index  

    # 打印有标签和无标签样本的数量
    print("Total training samples:", train_size)
    print("Number of labeled samples:", len(labeled_indices))
    print("Number of unlabeled samples:", len(unlabeled_indices))

    ## 获取剩下的样本作为无标签样本
    #unlabeled_indices = np.setdiff1d(train_index, labeled_indices)


    nus_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    nus_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = txt_set.shape[1]



#与MIRFlickr类基本类似
    class NUSWIDE(torch.utils.data.Dataset):

        def __init__(self, transform=None, target_transform=None, train=True, database=False,labeled=False):
            self.transform = transform
            self.target_transform = target_transform
            self.labeled =labeled
            if train:
                self.train_labels = label_set[train_index]
                self.train_index = train_index
                self.txt = txt_set[train_index]


                if labeled:
                    # 选择 30% 有标签数据
                    self.train_index = labeled_indices
                    self.train_labels = label_set[labeled_indices]
                    self.txt = txt_set[labeled_indices]
                else:
                    # 无标签数据为全部训练集
                    self.train_index = unlabeled_indices
                    self.train_labels = label_set[unlabeled_indices]
                    self.txt = txt_set[unlabeled_indices]


            elif database:
                self.train_labels = label_set[indexDatabase]
                self.train_index = indexDatabase
                self.txt = txt_set[indexDatabase]
            else:
                self.train_labels = label_set[indexTest]  
                self.train_index = indexTest
                self.txt = txt_set[indexTest]

        def __getitem__(self, index):

            nuswide = h5py.File(settings.IMG_DIR, 'r', libver='latest', swmr=True)
            #train_labels已经在__init__函数中进行了整理
            img, target = nuswide['IAll'][self.train_index[index]], self.train_labels[index]
            img = Image.fromarray(np.transpose(img, (2, 1, 0)))
            nuswide.close()
            
            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)