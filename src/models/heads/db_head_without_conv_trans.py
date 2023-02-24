# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:54
# @Author  : zhoujun
import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub


class Binarize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        bias = False
        self.conv = nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.sigmoid(x)

        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv', 'bn', 'relu'], inplace=True)


class DBHeadSimpleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = Binarize(in_channels)
        self.binarize.apply(self.weights_init)

        self.thresh = Binarize(in_channels)
        self.thresh.apply(self.weights_init)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        if self.training:
            shrink_maps = self.binarize(x)
            threshold_maps = self.thresh(x)
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = self.binarize(x)
        y = self.dequant(y)
        return y

    def fuse_model(self):
        for m in self.modules():
            if type(m) == Binarize:
                m.fuse_model()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
