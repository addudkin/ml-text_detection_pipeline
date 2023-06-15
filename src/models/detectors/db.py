# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun
# from addict import Dict
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.backbones import build_backbone
from src.models.necks import build_neck
from src.models.heads import build_head


class DB(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        # model_config = Dict(model_config)
        backbone_type = cfg.model.backbone.type
        neck_type = cfg.model.neck.type
        head_type = cfg.model.head.type
        self.backbone = build_backbone(backbone_type, **cfg.model.backbone.args)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **cfg.model.neck.args)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **cfg.model.head.args)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


class DBCism(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        # model_config = Dict(model_config)
        backbone_type = cfg.model.backbone.type
        neck_type = cfg.model.neck.type
        head_type = cfg.model.head.type
        self.backbone = build_backbone(backbone_type, **cfg.model.backbone.args)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **cfg.model.neck.args)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **cfg.model.head.args)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        ### Доработка для CISM возвращаем 2 таргета
        y_even, y_odd = self.head(neck_out)

        y_even = F.interpolate(y_even, size=(H, W), mode='bilinear', align_corners=True)
        y_odd = F.interpolate(y_odd, size=(H, W), mode='bilinear', align_corners=True)

        return y_even, y_odd


class DBCismEvenOddLines(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        # model_config = Dict(model_config)
        backbone_type = cfg.model.backbone.type
        neck_type = cfg.model.neck.type
        head_type = cfg.model.head.type
        self.backbone = build_backbone(backbone_type, **cfg.model.backbone.args)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **cfg.model.neck.args)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **cfg.model.head.args)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)

        ### Доработка для CISM возвращаем 2 таргета
        y_even, y_odd = self.head(neck_out)

        y_even = F.interpolate(y_even, size=(H, W), mode='bilinear', align_corners=True)
        y_odd = F.interpolate(y_odd, size=(H, W), mode='bilinear', align_corners=True)

        return y_even, y_odd


if __name__ == '__main__':
    import torch

    device = torch.device('cpu')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'resnest50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = Model(model_config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model.name)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')

