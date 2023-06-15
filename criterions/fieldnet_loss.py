import torch

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from segmentation_models_pytorch.losses import *

__all__ = [DiceLoss,
           JaccardLoss,
           LovaszLoss,
           FocalLoss,
           TverskyLoss,
           SoftBCEWithLogitsLoss,
           SoftCrossEntropyLoss,
           CrossEntropyLoss,
           BCEWithLogitsLoss]


class MultiLoss(torch.nn.Module):
    def __init__(self,
                 loss_funcs: dict):
        super().__init__()

        self.weights = []
        self.list_losses = []

        params = []

        for loss_func in loss_funcs.keys():
            params.append(loss_funcs[loss_func]['param'])
            self.list_losses.append(loss_funcs[loss_func]['name'])
            self.weights.append(loss_funcs[loss_func]['weight'])

        for loss_name, param in zip(self.list_losses, params):
            setattr(self, loss_name, eval(loss_name)(**param))

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> int:

        union_loss = 0
        for weight, loss_name in zip(self.weights, self.list_losses):
            union_loss += weight * getattr(self, loss_name)(predictions, targets)
        return union_loss
