import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from util import *

import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map#.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def IOU(pred, target):
    #inter = torch.sum(target * pred * mask, dim=(1, 2, 3))
    #union = torch.sum((target + pred) * mask, dim=(1, 2, 3)) - inter
    #iou_loss = 1 - (inter / union).mean()
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()


def IOU2(pred, target):
    #inter = torch.sum(target * pred * mask, dim=(1, 2, 3))
    #union = torch.sum((target + pred) * mask, dim=(1, 2, 3)) - inter
    #iou_loss = 1 - (inter / union).mean()
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss


def bce_ssim_loss(pred, gt):
    mask = (1 - ((gt > 0.4) * (gt < 0.6)).float()).detach() #((gt > 0.3) * (gt < 0.7)).float().detach()
    
    target = gt.gt(0.5).float()
    # print(torch.max(pred), torch.max(target))

    bce_out = nn.BCELoss(reduction='none')(pred, target)
    ssim_out = (1 - SSIM(window_size=11, size_average=False)(pred, target)).mean()
    # ssim_out = 0
    iou_out = IOU(pred, target)
    # iou_out = IOU_mask(pred, target, mask)

    # print(bce_out.shape, ssim_out.shape, iou_out.shape, mask.shape)

    # print((bce_out * mask).mean(), (ssim_out * mask).mean(), iou_out)
    # loss = (bce_out * mask).mean() + (ssim_out * mask).mean() + iou_out
    loss = bce_out.mean() + ssim_out.mean() + iou_out

    return loss


def bce_layer_loss(pred, gt, vars):
    target = gt.gt(0.5).float()

    bce_out = nn.BCELoss(reduction='none')(pred, target)
    ssim_out = (1 - SSIM(window_size=11, size_average=False)(pred, target)).mean()
    iou_out = IOU2(pred, target)
    exp_var = torch.exp(-vars * 20)  # 20表示PIXEL_WEIGHT

    loss = (bce_out * exp_var).mean() + (ssim_out * exp_var).mean() + (iou_out * exp_var).mean()

    return loss


def get_contour(label):
    lbl = label.gt(0.5).float()
    ero = 1 - F.max_pool2d(1 - lbl, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(lbl, kernel_size=5, stride=1, padding=2)  # dilation

    edge = dil - ero
    return edge


# Boundary-aware Texture Matching Loss
def BTMLoss(pred, image, radius):
    alpha = 200
    modal = 'c'
    num_modal = len(modal) if 'c' in modal else len(modal) + 1  # 计算模态数

    slices = range(0, 1 * num_modal + 1, 1)  # 其中1表示图像通道数
    # pred:[1, 1, 576, 768] image:[1, 2, 576, 768]
    sal_map = F.interpolate(pred, scale_factor=0.25, mode='bilinear', align_corners=True)
    image_ = F.interpolate(image, size=sal_map.shape[-2:], mode='bilinear', align_corners=True)
    mask = get_contour(sal_map)  # 通过预测得到的掩码
    features = torch.cat([image_, sal_map], dim=1)

    N, C, H, W = features.shape  # features:[1, 3, 144, 192]

    diameter = 2 * radius + 1
    # kernels:[1, 3, 11, 11, 144, 192]
    kernels = F.unfold(features, diameter, 1, radius).view(N, C, diameter, diameter, H, W)
    # kernels:[1, 3, 11, 11, 144, 192]
    kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)  # 计算纹理向量
    dis_modal = 1
    for idx, slice in enumerate(slices):
        if idx == len(slices) - 1:
            continue
        dis_map = (-alpha * kernels[:, slice:slices[idx + 1]] ** 2).sum(dim=1, keepdim=True).exp()
        # Only RGB
        # if config['only_rgb'] and idx > 0:
        #     dis_map = dis_map * 0 + 1
        dis_modal = dis_modal * dis_map  # 表示Ta

    dis_sal = torch.abs(kernels[:, slices[-1]:])  # 表示Ts
    distance = dis_modal * dis_sal  # Ts 和 Ta相乘
    loss = distance.view(N, 1, (radius * 2 + 1) ** 2, H, W).sum(dim=2)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def Loss(preds, target, var, config):
    loss = 0
    ws = [1, 0.2]
    for pred, w in zip(preds['sal'], ws):
        pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        # loss += bce_ssim_loss(torch.sigmoid(pred), target) * w
        loss += bce_layer_loss(torch.sigmoid(pred), target, var) * w

        # ac_loss = BTMLoss(pred, priors, 5)  # Texture 损失
        # loss += 0.005 * ac_loss
        
    return loss