import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import *
from ptsemseg.lovasz_losses import *


def cross_entropy2d(input, target, weight=None, size_average=True, ignore_index=250):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsqueeze(1)
        target = F.interpolate(target.float(), size=(h, w), mode='nearest').long()
        target = target.squeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear', align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=ignore_index,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss


def focal_loss(input, target, weight=None, size_average=True, ignore_index=250, alpha=0.25, gamma=2.0):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsqueeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.squeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear', align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) # size: (n*h*w, c)
    target = target.view(-1) # size: (n*h*w,)

    mask = (target >= 0) * (target != ignore_index)
    input = input[mask, :]
    target = target[mask]
    target = target.view(-1, 1) # size: (n*h*w, 1)

    target = torch.zeros(input.size(), device=torch.device('cuda')).scatter_(1, target, 1.) # one-hot, size: (n*h*w, c)

    p = F.softmax(input, dim=1)
    log_p = F.log_softmax(input, dim=1)

    pt = (p * target).sum(1).view(-1, 1)
    log_pt = (log_p * target).sum(1).view(-1, 1)

    ce_loss = - log_pt
    loss = alpha * ce_loss # alpha-balanced CE
    loss = loss * torch.pow(1-pt, gamma) # with FL
    loss = loss.sum()

    if size_average:
        loss /= mask.data.sum().float()
    return loss


def dice_loss(input, target, smooth=1.):
    input_flat = input.contiguous().float().view(-1)
    target_flat = target.contiguous().float().view(-1)

    product = input_flat * target_flat
    intersection = product.sum()
    coefficient = (2.*intersection + smooth) / (input_flat.sum()+target_flat.sum() + smooth)
    loss = 1. - coefficient #- torch.log(coefficient)
    return loss


def seg_loss(input, target, weight=None, size_average=True, scale_weight=1.0, lambda_ce=1.0, lambda_fl=1.0, lambda_dc=1.0, lambda_lv=1.0):
    ce_loss = cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average) if lambda_ce > 0.0 else 0.0
    fl_loss = focal_loss(input, target, alpha=10.0, gamma=2.0) if lambda_fl > 0.0 else 0.0
    dc_loss = dice_loss(F.softmax(input, dim=1)[:, 1, :, :], target) if lambda_dc > 0.0 else 0.0
    lv_loss = lovasz_softmax(F.softmax(input, dim=1), target, only_present=False, per_image=True, ignore=250) if lambda_lv > 0.0 else 0.0

    loss = scale_weight * (lambda_ce * ce_loss + lambda_fl * fl_loss + lambda_dc * dc_loss + lambda_lv * lv_loss)
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None, lambda_ce=1.0, lambda_fl=1.0, lambda_dc=1.0, lambda_lv=1.0):
    if not isinstance(input, tuple):
        return seg_loss(input, target, weight=weight, size_average=size_average, lambda_ce=lambda_ce, lambda_fl=lambda_fl, lambda_dc=lambda_dc, lambda_lv=lambda_lv)

    n_inp = len(input)
    # Auxiliary training for PSPNet [1.0, 0.4]
    if scale_weight is None: # scale_weight: torch tensor type
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp, device=torch.device('cuda')), torch.arange(n_inp, device=torch.device('cuda')).float())

    loss = 0.0
    for i in range(n_inp):
        if isinstance(input[i], tuple):
            n_j = len(input[i])
            for j in range(n_j):
                loss = loss + seg_loss(input[i][j], target, weight=weight, size_average=size_average, scale_weight=scale_weight[i], lambda_ce=lambda_ce, lambda_fl=lambda_fl, lambda_dc=lambda_dc, lambda_lv=lambda_lv) / n_j
        else:
            loss = loss + seg_loss(input[i], target, weight=weight, size_average=size_average, scale_weight=scale_weight[i], lambda_ce=lambda_ce, lambda_fl=lambda_fl, lambda_dc=lambda_dc, lambda_lv=lambda_lv)
    return loss
