import torch.nn as nn
import mmcv

from mmocr.models.textdet.necks import FPNC
from mmdet.models import ResNet

from models.heads import build_head
from omegaconf import OmegaConf


class DBNetpp(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = ResNet(
            **OmegaConf.to_container(config['model']['backbone']['params'])
        )
        self.neck = FPNC(
            **OmegaConf.to_container(config['model']['neck']['params'])
        )

        self.head = build_head(config['model']['head']['type'], in_channels=256, **config['model']['head']['params'])

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward(self, img, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        preds = self.head(x)
        return preds
