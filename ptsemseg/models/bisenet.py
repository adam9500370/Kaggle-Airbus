import torch
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import *
from ptsemseg.loss import *

class bisenet(nn.Module):
    def __init__(self, n_classes=2, is_batchnorm=True):
        super(bisenet, self).__init__()

        bias = not is_batchnorm

        # BiSeNet
        self.spatial_path = spatialPath(is_batchnorm=is_batchnorm)
        self.context_path = contextPath(is_batchnorm=is_batchnorm)
        self.ffm3 = featureFusionModule(in_channels=128, out_channels=32, is_batchnorm=is_batchnorm)
        self.ffm2 = featureFusionModule(in_channels=64, out_channels=32, is_batchnorm=is_batchnorm)
        self.ffm1 = featureFusionModule(in_channels=64, out_channels=32, is_batchnorm=is_batchnorm)

        # (Auxiliary) Classifier
        self.aux_cls_32x = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
        self.aux_cls_ff3 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)
        self.cls_ff1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, padding=0)

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d

    def forward(self, x, output_size=(768, 768)):
        x_s3, x_s2, x_s1 = self.spatial_path(x)
        x_c, x_32x = self.context_path(x)
        x_ff3 = self.ffm3(x_s3, x_c)
        if self.training:
            out_aux_32x = self.aux_cls_32x(x_32x)
            out_aux_32x = F.interpolate(out_aux_32x, size=output_size, mode='bilinear', align_corners=True)

            out_aux_ff3 = self.aux_cls_ff3(x_ff3)
            out_aux_ff3 = F.interpolate(out_aux_ff3, size=output_size, mode='bilinear', align_corners=True)

        x_ff3 = F.interpolate(x_ff3, size=x_s2.shape[2:], mode='bilinear', align_corners=True)
        x_ff2 = self.ffm2(x_s2, x_ff3)

        x_ff2 = F.interpolate(x_ff2, size=x_s1.shape[2:], mode='bilinear', align_corners=True)
        x_ff1 = self.ffm1(x_s1, x_ff2)

        out = self.cls_ff1(x_ff1)
        out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=True)

        if self.training:
            return (out, out_aux_ff3, out_aux_32x)
        else: # eval mode
            return out
