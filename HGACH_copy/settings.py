import logging
import time
import os.path as osp

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False
# EVAL = True
# dataset can be 'MIRFlickr' or 'NUSWIDE'

DATASET = 'MIRFlickr'


if DATASET == 'MIRFlickr':

    LABEL_DIR = './datasets/mirflickr/mirflickr25k-lall.mat'
    TXT_DIR = './datasets/mirflickr/mirflickr25k-yall.mat'
    IMG_DIR = './datasets/mirflickr/mirflickr25k-iall.mat'
    
    BETA = 0.9
    LAMBDA1 = 0.1
    LAMBDA2 = 0.1
    LAMBDA3 = 1 
    NUM_EPOCH = 500
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 5

    alpha = 1.0
    beta = 4.0

    threshold = 0.33 #原始
    



    K =5

#################
    A =0.5
    B =0.1
    C =0.4



###############MITHloss:半监督
    INTRA = 2  # 通道内损失的权重
    INTER = 2  # 通道间损失的权重

    D = 3.0
    E = 2.0





if DATASET == 'NUSWIDE':
    LABEL_DIR = './datasets/NUSWIDE/nus-wide-tc21-lall.mat'
    TXT_DIR = './datasets/NUSWIDE/nus-wide-tc21-yall.mat'
    IMG_DIR = './datasets/NUSWIDE/IAll/nus-wide-tc21-iall.mat'


    LAMBDA1 = 0.3
    LAMBDA2 = 0.3
    LAMBDA3 = 1
    NUM_EPOCH = 90
    LR_IMG = 0.001
    LR_TXT = 0.01
    EVAL_INTERVAL = 5


    threshold = -0.3   #高斯阈值 th

    alpha = 1.0    #高斯相异阈值参数 
    beta = 4.0   #高斯相似阈值参数

    K = 3

    ##################初始相似度矩阵权重
    A =0.4
    B =0.3
    C =0.3





    ###############半监督
    INTRA = 2  # 通道内损失的权重
    INTER = 2  # 通道间损失的权重

    D = 3.0    #图像模态内损失权重
    E = 2.0    #文本模态内损失权重






BATCH_SIZE =64
CODE_LEN =32

l4 = 0.3
l5 = 0.0005
sim1 = 1.4
nnk = 0.08
temperature = 1.0

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
GPU_ID = 0
NUM_WORKERS = 8
EPOCH_INTERVAL = 2

MODEL_DIR = './HGACH/checkpoint'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './HGACH/log/log_f25k'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)