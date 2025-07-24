#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch
from torch.nn.modules.utils import _pair, _quadruple
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import ot
def smaller(x1, x2, min_delta=0):
    return x2 - x1 > min_delta


def bigger(x1, x2, min_delta=0):
    return x1 - x2 > min_delta


class ModelSaver:

    def __init__(self, get=lambda trainer: trainer.state["val_loss"], better=smaller, path=None):

        self.get = get
        self.better = better
        self.path = path

        self.best = None

    def save(self, trainer):
        trainer.save(path=None, value=self.best)

    def check(self, trainer):

        res = self.get(trainer)

        if self.best is None:
            self.best = res
            self.save(trainer)

        else:
            if self.better(res['abs'], self.best['abs']):
                self.save(trainer)
                self.best = res

    def register(self, callback_dict):
        callback_dict["after_epoch"].append(lambda trainer: self.check(trainer))


def move_to(list, device):
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in list]


def median_pool(x, kernel_size=3, stride=1, padding=0):
    k = _pair(kernel_size)
    stride = _pair(stride)
    padding = _quadruple(padding)

    x = F.pad(x, padding, mode='reflect')
    x = x.unfold(2, k[0], stride[0]).unfold(3, k[1], stride[1])
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]

    return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: 形状为 (b, 2, 128, 128) 的预测张量
        # target: 形状为 (b, 1, 128, 128) 的目标张量
        pred = pred.permute(0, 2, 3, 1)  # 形状变为 (b, 128, 128, 2)
        pred = pred.contiguous().view(-1, 2)  # 形状变为 (b*128*128, 2)

        target = target.reshape(-1, 1)  # 使用 reshape 而不是 view
        target = target.long()

        logpt = F.log_softmax(pred, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_gausian_filter(sigma, sz):
    xpos, ypos = torch.meshgrid(torch.arange(sz), torch.arange(sz))
    output = torch.ones([sz, sz, 1, 1])
    midpos = sz // 2
    d = (xpos-midpos)**2 + (ypos-midpos)**2
    gauss = torch.exp(-d / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return gauss

def gaussian_filter_(img, n, sigma):
    """
    img: image tensor of size (1, 1, height, width)
    n: size of the Gaussian filter (n, n)
    sigma: standard deviation of the Gaussian distribution
    """
    # Create a Gaussian filter
    gaussian_filter1 = get_gausian_filter(sigma, n)
    # Add extra dimensions for the color channels and batch size
    gaussian_filter1 = gaussian_filter1.view(1, 1, n, n)
    gaussian_filter1 = gaussian_filter1.to(img.device)
    # Perform 2D convolution
    filtered_img = F.conv2d(img, gaussian_filter1, padding=n//2)
    return filtered_img

def Dp(image, sigma, patch_size, xshift, yshift):
    shift_image = torch.roll(image, shifts=(xshift, yshift), dims=(-1, -2))
    diff = image - shift_image
    diff_square = diff ** 2
    res = gaussian_filter_(diff_square, patch_size, sigma)
    return res

def mind(image, sigma=2.0, eps=1e-5, neigh_size=9, patch_size=7):
    reduce_size = (patch_size + neigh_size - 2) // 2
    # estimate the local variance of each pixel within the input image.
    Vimg = Dp(image, sigma, patch_size, -1, 0) + Dp(image, sigma, patch_size, 1, 0) + \
            Dp(image, sigma, patch_size, 0, -1) + Dp(image, sigma, patch_size, 0, 1)
    Vimg = Vimg / 4 + eps * torch.ones_like(Vimg)

    # estimate the (R*R)-length MIND feature by shifting the input image by R*R times.
    xshift_vec = np.arange(-neigh_size//2, neigh_size - neigh_size // 2)
    yshift_vec = np.arange(-neigh_size// 2, neigh_size - neigh_size // 2)

    #print(xshift_vec, yshift_vec)

    iter_pos = 0
    for xshift in xshift_vec:
        for yshift in yshift_vec:
            if (xshift,yshift) == (0,0):
                continue
            MIND_tmp = torch.exp(-Dp(image, sigma, patch_size, xshift, yshift) / Vimg) # MIND_tmp : 1x1x256x256
            tmp = MIND_tmp[...,reduce_size:-reduce_size, reduce_size:-reduce_size,None] # 1x1x250x250x1
            output = tmp if iter_pos == 0 else torch.cat((output,tmp), -1)
            iter_pos += 1

    # normalization.
    output = torch.divide(output, torch.max(output, dim=-1, keepdim=True)[0])

    return output


def wasserstein_loss(x, y):
    # 将张量转换为 numpy 数组
    x = x.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()

    # 计算直方图
    x_hist, _ = np.histogram(x, bins=300, range=(0, 1.5), density=True)
    y_hist, _ = np.histogram(y, bins=300, range=(0, 1.5), density=True)

    # 创建均匀分布的权重
    weights_x = np.ones_like(x_hist) / len(x_hist)
    weights_y = np.ones_like(y_hist) / len(y_hist)

    # 创建 cost matrix
    M = ot.dist(x_hist.reshape(-1, 1), y_hist.reshape(-1, 1))

    # 计算 Wasserstein 距离
    wasserstein_dist = ot.emd2(weights_x, weights_y, M)
    return torch.tensor(wasserstein_dist)


def similarity_loss(target, pred):
    """
    计算目标张量和预测张量之间的相似性约束损失。

    参数:
    target -- 目标张量，形状为 (batch_size, channels, height, width)
    pred -- 预测张量，形状为 (batch_size, channels, height, width)

    返回:
    loss -- 相似性约束损失
    """
    # 将张量展平成 (batch_size, channels, -1) 的形状
    target_flat = target.view(target.size(0), target.size(1), -1)
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)

    # 计算余弦相似性
    cosine_similarity = F.cosine_similarity(target_flat, pred_flat, dim=-1)

    # 计算相似性损失
    loss = 1 - cosine_similarity.mean()

    return loss

def kl_divergence(p, q):
    p = torch.softmax(p, dim=1)
    q = torch.softmax(q, dim=1)
    res = torch.sum(p * torch.log((p + 1e-10) / (q + 1e-10)), dim=1)
    return res


class MINDLoss(nn.Module):
    def __init__(self,
                 sigma=2.0,
                 eps=1e-5,
                 neigh_size=9,
                 patch_size=7):
        super(MINDLoss, self).__init__()
        self.sigma = sigma
        self.eps = eps
        self.neigh_size = neigh_size
        self.patch_size = patch_size

    def forward(self, pred, gt):
        pred_mind = mind(pred, self.sigma, self.eps, self.neigh_size, self.patch_size)
        gt_mind = mind(gt, self.sigma, self.eps, self.neigh_size, self.patch_size)

        return pred_mind - gt_mind
def BinaryDiceLoss(pred, target, smooth=1e-6):
    # pred: 形状为 (b, 2, 128, 128) 的预测张量
    # target: 形状为 (b, 1, 128, 128) 的目标张量
    pred = pred[:, 1, :, :]  # 取出第2类的概率，形状为 (b, 128, 128)
    target = target.squeeze(1)  # 形状变为 (b, 128, 128)

    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)

    return 1 - dice.mean()

class CustomInfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(CustomInfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, smallb):
        embeddings = F.normalize(embeddings, dim=-1)
        batch_size = embeddings.shape[0]
        k = batch_size // smallb
        loss = 0
        for i in range(k):
            pos_indices = [i + index * k for index in range(1, smallb)]
            neg_indices = [j for j in range(batch_size) if j not in pos_indices and j != i]

            pos_samples = embeddings[pos_indices] # (a, c, dims)
            neg_samples = embeddings[neg_indices] # (b, c, dims)

            query = embeddings[i].unsqueeze(0)  # (1, c, dims)

            pos_sim = torch.einsum('ncd,mcd->nm', query, pos_samples) / self.temperature
            neg_sim = torch.einsum('ncd,mcd->nm', query, neg_samples) / self.temperature
            pos_sim = torch.mean(pos_sim, dim=0, keepdim=True)
            logits = torch.cat([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(pos_sim.shape[0], dtype=torch.long).to(query.device)

            loss += F.cross_entropy(logits, labels)

        loss /= batch_size
        return loss


class STDLayer(nn.Module):

    def __init__(
            self,
            nb_classes,
            nb_iterations=10,
            nb_kerhalfsize=3,
    ):
        """
        :param nb_classes: number of classes
        :param nb_iterations: iterations number
        :param nb_kerhalfsize: the half size of neigbourhood
        """
        super(STDLayer, self).__init__()

        self.nb_iterations = nb_iterations
        self.nb_classes = nb_classes
        self.ker_halfsize = nb_kerhalfsize

        # Learnable version: sigma of Gasussian function; entropic parameter epsilon; regularization parameter lam
        self.nb_sigma = nn.Parameter(torch.FloatTensor([10.0] * nb_classes).view(nb_classes, 1, 1))
        self.entropy_epsilon = nn.Parameter(torch.FloatTensor([1.0]))
        self.lam = nn.Parameter(torch.FloatTensor([5.0]))

        # Fixed parmaters.
        # self.nb_sigma = Variable(torch.FloatTensor([10.0]*nb_classes).view(nb_classes,1,1),requires_grad=False).cuda()
        # self.entropy_epsilon=Variable(torch.FloatTensor([1.0]),requires_grad=False).cuda()
        # self.lam=Variable(torch.FloatTensor([5.0]),requires_grad=False).cuda()

        # softmax
        self.softmax = nn.Softmax2d()

    def forward(self, o):
        u = self.softmax(o * (self.entropy_epsilon ** 2.0))
        # std kernel
        ker = STDLayer.STD_Kernel(self.nb_sigma, self.ker_halfsize)
        # main iteration
        for i in range(self.nb_iterations):
            # 1. subgradient
            q = F.conv2d(1.0 - 2.0 * u, ker, padding=self.ker_halfsize, groups=self.nb_classes)
            # 2. Softmax
            u = self.softmax((o - self.lam * q) * (self.entropy_epsilon ** 2.0))

        return u

    def STD_Kernel(sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        x = x.to(sigma.device)
        y = y.to(sigma.device)
        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2) / (2.0 * sigma ** 2))
        ker = ker / (ker.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-15)
        ker = ker.unsqueeze(1)
        return ker




import random
import math
import imgaug.augmenters as iaa
import numpy as np

perlin_scale = 3
min_perlin_scale = 1
perlin_noise_threshold: float = 0.3
def generate_perlin_noise_mask(size, perlin_scale, min_perlin_scale) -> np.ndarray:
    # define perlin noise scale
    # self.perlin_scale = random.randint(3, 5)
    # self.min_perlin_scale = random.randint(1, 2)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    # generate perlin noise
    perlin_noise = rand_perlin_2d_np((size[0], size[1]), (perlin_scalex, perlin_scaley))

    # apply affine transform
    rot = iaa.Affine(rotate=(-90, 90))
    perlin_noise = rot(image=perlin_noise)

    # make a mask by applying threshold
    mask_noise = np.where(
        perlin_noise > perlin_noise_threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise)
    )
    return mask_noise

def generate_perlin_noise(img, perlin_scale, min_perlin_scale):
    b, c, h, w = img.shape
    mask = img.sum(dim=1, keepdim=True) > 0.01
    res = []
    for i in range(b*c):
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((h, w), (perlin_scalex, perlin_scaley))

        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        res.append(torch.from_numpy(perlin_noise))
    # make a mask by applying threshold
    return torch.stack(res, dim=0).view(b, c, h, w).float().to(img.device)* mask + img

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    def lerp_np(x, y, w):
        fin_out = (y - x) * w + x
        return fin_out

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

import numpy as np
import math


def perlin_noise(imgs):
    res = []
    for img in imgs:
        target_foreground_mask = img.sum(dim=0) > 0.01
        target_foreground_mask = target_foreground_mask.to('cpu')
        perlin_noise_mask = torch.from_numpy(generate_perlin_noise_mask(img.shape[-2:]))

        anomaly_mask = perlin_noise_mask * target_foreground_mask
        while anomaly_mask.sum() < 5 and target_foreground_mask.sum() > 100:
            perlin_noise_mask = torch.from_numpy(generate_perlin_noise_mask(img.shape[-2:]))
            ## mask
            anomaly_mask = perlin_noise_mask * target_foreground_mask
        anomaly_mask = anomaly_mask.to(img.device)
        # step 2. generate texture or structure anomaly
        index_list = list(range(img.shape[0]))
        anomaly_source_img = img.clone()
        factor = [random.uniform(0.75, 1.5), random.uniform(0.25, 1.25), random.uniform(0.25, 1.25),
                  random.uniform(0.75, 1.5)]
        for i in range(img.shape[0]):
            a = i
            k = random.uniform(0.75, 1)
            while a == i:
                a = random.choice(index_list)
            anomaly_source_img[i] = factor[i] * (k * img[a] + (1 - k) * img[i]) * anomaly_mask + img[i] * (
                    1 - anomaly_mask)
        res.append(anomaly_source_img)
    return torch.stack(res, dim=0)
if __name__ == '__main__':
    a = torch.randn((2, 128, 128))
    b = torch.randn((2, 128, 128))
    mind_loss = MINDLoss()
    b = mind_loss(a, b)
    print(b)