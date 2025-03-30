import numpy as np
import settings
import torch
import matplotlib.pyplot as plt
# from settings import a, b,c
from scipy.stats import norm

def build_G_from_S(S, k):
    # S: similarity matrix
    # k: number of nearest neighbors
    # G: graph
    G = torch.ones(S.shape).cuda() * -1.5
    # G = torch.zeros(S.shape).cuda()
    G_ = torch.where(S > settings.threshold, S, -1.5).cuda()

    for i in range(G_.shape[0]):
        idx = torch.argsort(-G_[i])[:k]
        G[i][idx] = G_[i][idx]
    del G_
    torch.cuda.empty_cache()
    return G

def generate_robust_S(s, alpha, beta):
    """
    Generate robust similarity matrix and save histograms to specified path.

    :param s: Similarity matrix
    :param alpha: Parameter for positive robustness
    :param beta: Parameter for negative robustness
    """

     # 如果 S 是 PyTorch 张量，则转换为 NumPy 数组
    if isinstance(s, torch.Tensor):
        s = s.cpu().numpy()  # 将张量转换为 NumPy 数组
    # Define the save path
    save_path = './HGACH/draw_image'

    # Ensure the save path exists
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    S = s
    max_count = 0
    max_cos = 0

    interval = 1 / 1000
    cur = -1.0
    for i in range(2000):
        cur_cnt = np.sum((S > cur) & (S < cur + interval))
        if max_count < cur_cnt:
            max_count = cur_cnt
            max_cos = cur
        cur += interval

    # Split positive and negative similarity matrix
    flat_S = S.reshape((-1, 1))
    left = flat_S[np.where(flat_S <= max_cos)[0]]
    right = flat_S[np.where(flat_S >= max_cos)[0]]

    # Reconstruct
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([max_cos - np.maximum(right - max_cos, max_cos - right), right])

    # Fit to Gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    # print('left mean: ', left_mean)
    # print('left std: ', left_std)
    # print('threshold:', left_mean - alpha * left_std)
    # print('right mean: ', right_mean)
    # print('right std: ', right_std)
    # print('threshold:', right_mean + beta * right_std)

    S = np.where(S >= right_mean + beta * right_std, 1, S)
    #S = np.where(S <= left_mean - alpha * left_std, 0, S)
    S = np.where(S <= left_mean - alpha * left_std, -1, S)

    # Do not display the y-axis
    plt.gca().axes.get_yaxis().set_visible(False)

    # Draw the histogram and save the plots
    plt.hist(left, bins=10000, density=True, alpha=0.6, color='g')
    plt.savefig(os.path.join(save_path, 'left.png'), format='png')
    plt.close()

    plt.hist(right, bins=10000, density=True, alpha=0.6, color='r')
    plt.savefig(os.path.join(save_path, 'right.png'), format='png')
    plt.close()

    plt.hist(flat_S, bins=10000, density=True, alpha=0.6, color='b')
    plt.savefig(os.path.join(save_path, 'flat_S.png'), format='png')
    plt.close()

    return S
    # generate robust similarity matrix

    # show the fitting result
    # plt.hist(left, bins=100, density=True, alpha=0.6, color='g')
    # plt.hist(right, bins=100, density=True, alpha=0.6, color='r')
    # plt.show()
    # plt.close()
