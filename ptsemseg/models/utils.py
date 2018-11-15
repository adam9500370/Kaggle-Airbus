import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, groups=1, is_batchnorm=True):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, groups=1, negative_slope=0.0, is_batchnorm=True):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)
        relu_mod = nn.LeakyReLU(negative_slope=negative_slope, inplace=True) if negative_slope > 0.0 else nn.ReLU(inplace=True)

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          relu_mod,)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          relu_mod,)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, is_batchnorm=True):
        super(deconv2DBatchNorm, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                      padding=padding, stride=stride, bias=bias)

        if is_batchnorm:
            self.dcb_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),)
        else:
            self.dcb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, negative_slope=0.0, is_batchnorm=True):
        super(deconv2DBatchNormRelu, self).__init__()

        conv_mod = nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                      padding=padding, stride=stride, bias=bias)
        relu_mod = nn.LeakyReLU(negative_slope=negative_slope, inplace=True) if negative_slope > 0.0 else nn.ReLU(inplace=True)

        if is_batchnorm:
            self.dcbr_unit = nn.Sequential(conv_mod,
                                           nn.BatchNorm2d(int(n_filters)),
                                           relu_mod,)
        else:
            self.dcbr_unit = nn.Sequential(conv_mod,
                                           relu_mod,)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class pyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, is_batchnorm=True, mode='cat'): # PSPNet: 'cat'; ICNet: 'sum'
        super(pyramidPooling, self).__init__()

        bias = not is_batchnorm

        self.mode = mode
        self.pool_sizes = pool_sizes

        if mode == 'cat':
            self.paths = []
            for i in range(len(pool_sizes)):
                self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, is_batchnorm=is_batchnorm))
            self.path_module_list = nn.ModuleList(self.paths)

    def forward(self, x):
        h, w = x.shape[2:]

        strides = [(int(h/pool_size), int(w/pool_size)) for pool_size in self.pool_sizes]
        k_sizes = [(int(h - strides[i][0]*(pool_size-1)), int(w - strides[i][1]*(pool_size-1))) for i, pool_size in enumerate(self.pool_sizes)]

        if self.mode == 'cat':
            output_slices = [x]
            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
                output_slices.append(out)
            return torch.cat(output_slices, dim=1)
        else: #self.mode == 'sum'
            output = 0
            for i, pool_size in enumerate(self.pool_sizes):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
                output = output + out
            return output


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
                 stride, dilation=1, groups=1, is_batchnorm=True):
        super(bottleNeckPSP, self).__init__()

        bias = not is_batchnorm

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                        stride=stride, padding=dilation,
                                        bias=bias, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias, is_batchnorm=is_batchnorm)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x) if self.in_channels != self.out_channels else x
        return F.relu(conv+residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, groups=1, is_batchnorm=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                        stride=1, padding=dilation,
                                        bias=bias, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, is_batchnorm=is_batchnorm)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x+residual, inplace=True)


class residualBlockPSP(nn.Module):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1, groups=1, is_batchnorm=True):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        layers = []
        layers.append(bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm))
        for i in range(n_blocks-1):
            layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation=dilation, groups=groups, is_batchnorm=is_batchnorm))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class contextPath(nn.Module):
    def __init__(self, is_batchnorm=True):
        super(contextPath, self).__init__()

        bias = not is_batchnorm

        # Encoder (context path)
        self.backbone = resnet18(pretrained=True)
        self.pyramid_pooling_module = pyramidPooling(512, [6, 3, 2, 1], mode='sum')

        self.arm_32x = attentionRefinementModule(512, is_batchnorm=is_batchnorm)
        self.arm_16x = attentionRefinementModule(256, is_batchnorm=is_batchnorm)
        self.arm_8x = attentionRefinementModule(128, is_batchnorm=is_batchnorm)

        # Decoder
        self.conv_32x = conv2DBatchNormRelu(in_channels=512, n_filters=256, k_size=3, stride=1, padding=1, bias=bias, is_batchnorm=is_batchnorm)
        self.de_res_blocks_16x = residualBlockPSP(n_blocks=3, in_channels=512, mid_channels=64, out_channels=128, stride=1, dilation=1)
        self.de_res_blocks_8x = residualBlockPSP(n_blocks=3, in_channels=256, mid_channels=32, out_channels=64, stride=1, dilation=1)

    def forward(self, x):
        feat_2x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        feat_4x = self.backbone.maxpool(feat_2x)
        feat_4x = self.backbone.layer1(feat_4x)
        feat_8x = self.backbone.layer2(feat_4x)
        feat_16x = self.backbone.layer3(feat_8x)
        feat_32x = self.backbone.layer4(feat_16x)
        feat_ppm = self.pyramid_pooling_module(feat_32x)

        feat_32x = self.arm_32x(feat_32x)
        feat_16x = self.arm_16x(feat_16x)
        feat_8x = self.arm_8x(feat_8x)

        feat_32x = feat_32x + feat_ppm
        feat_32x = self.conv_32x(feat_32x)

        up_feat_32x = F.interpolate(feat_32x, size=feat_16x.shape[2:], mode='bilinear', align_corners=True)
        feat_16x = torch.cat([feat_16x, up_feat_32x], dim=1)
        feat_16x = self.de_res_blocks_16x(feat_16x)

        up_feat_16x = F.interpolate(feat_16x, size=feat_8x.shape[2:], mode='bilinear', align_corners=True)
        feat_8x = torch.cat([feat_8x, up_feat_16x], dim=1)
        feat_8x = self.de_res_blocks_8x(feat_8x)

        return feat_8x, feat_32x


class spatialPath(nn.Module):
    def __init__(self, is_batchnorm=True):
        super(spatialPath, self).__init__()

        bias = not is_batchnorm

        self.cbr1 = conv2DBatchNormRelu(in_channels=3, n_filters=32, k_size=3, stride=2, padding=1, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr2 = conv2DBatchNormRelu(in_channels=32, n_filters=32, k_size=3, stride=2, padding=1, bias=bias, is_batchnorm=is_batchnorm)
        self.cbr3 = conv2DBatchNormRelu(in_channels=32, n_filters=64, k_size=3, stride=2, padding=1, bias=bias, is_batchnorm=is_batchnorm)

    def forward(self, x):
        x1 = self.cbr1(x)
        x2 = self.cbr2(x1)
        x3 = self.cbr3(x2)
        return x3, x2, x1


class attentionRefinementModule(nn.Module):
    def __init__(self, in_channels, is_batchnorm=True):
        super(attentionRefinementModule, self).__init__()

        bias = not is_batchnorm

        self.arm = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
                                 conv2DBatchNorm(in_channels, in_channels, k_size=1, stride=1, padding=0, bias=bias, dilation=1, is_batchnorm=is_batchnorm),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.arm(x)


class featureFusionModule(nn.Module):
    def __init__(self, in_channels=128, out_channels=32, reduction=2, is_batchnorm=True):
        super(featureFusionModule, self).__init__()

        bias = not is_batchnorm

        self.cbr = conv2DBatchNormRelu(in_channels=in_channels, n_filters=out_channels, k_size=3, stride=1, padding=1, bias=bias, is_batchnorm=is_batchnorm)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
                                 conv2DBatchNormRelu(out_channels, out_channels // reduction, k_size=1, stride=1, padding=0, bias=bias, dilation=1, is_batchnorm=is_batchnorm),#False),
                                 conv2DBatchNorm(out_channels // reduction, out_channels, k_size=1, stride=1, padding=0, bias=bias, dilation=1, is_batchnorm=is_batchnorm),#False),
                                 nn.Sigmoid())

    def forward(self, x1, x2):
        feat = torch.cat([x1, x2], dim=1)
        feat = self.cbr(feat)
        feat_cse = self.cse(feat)
        return feat + feat * feat_cse



def flip(x, dim): # tensor flip
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
