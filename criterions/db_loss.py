import torch
from torch import nn
from segmentation_models_pytorch.losses import *

from criterions.seg_losses import BalanceCrossEntropyLoss, DiceLoss, MaskL1Loss


class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta

        self.dice_loss = DiceLoss(eps=eps)
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, loss=self.dice_loss)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            binary_maps = pred[:, 2, :, :]
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            # alpha=1.0, beta=10
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics


class DBLossIntersection(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, gamma=5, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dice_loss = DiceLoss(eps=eps)
        self.focal_loss = FocalLoss(
            reduction="mean",
            mode="multilabel"
        )
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, loss=self.dice_loss)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        intersection_maps = pred[:, 3:, :, :].contiguous()

        shrink_map_target = batch['shrink_map'].clone()
        shrink_map_target[shrink_map_target==2] = 0

        intersection_map_target_1 = batch['shrink_map'].clone()

        intersection_map_target_1[intersection_map_target_1 == 1] = 0
        intersection_map_target_1[intersection_map_target_1 == 2] = 1

        intersection_map_target_0 = (intersection_map_target_1 == 0).float()

        intersection_targets = torch.stack((intersection_map_target_0, intersection_map_target_1), dim=1)

        loss_shrink_maps = self.bce_loss(shrink_maps, shrink_map_target, batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            binary_maps = pred[:, 2, :, :]
            loss_binary_maps = self.dice_loss(binary_maps, shrink_map_target, batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps

            focal_loss = self.focal_loss(intersection_maps, intersection_targets)
            metrics['loss_focal_intersec'] = focal_loss
            # alpha=1.0, beta=10
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps + self.gamma * focal_loss
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics


class DBLossV2(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta

        self.dice_loss = DiceLoss(eps=eps)
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, loss=self.dice_loss)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, preds, targets):

        metrics = {
            'loss_shrink_maps': 0,
            'loss_threshold_maps': 0,
            'loss_binary_maps': 0,
            'loss': 0,
        }

        for pred, batch, label in zip(preds, targets, ['even', 'odd']):
            shrink_maps = pred[:, 0, :, :]
            threshold_maps = pred[:, 1, :, :]

            loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
            loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])

            metrics['loss_shrink_maps'] += loss_shrink_maps
            metrics['loss_threshold_maps'] += loss_threshold_maps

            if pred.size()[1] > 2:
                binary_maps = pred[:, 2, :, :]
                loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
                metrics['loss_binary_maps'] += loss_binary_maps
                # alpha=1.0, beta=10
                loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
                metrics['loss'] += loss_all
                metrics[f"{label}_loss"] = loss_all
            else:
                metrics['loss'] += loss_shrink_maps
                metrics[f"{label}_loss"] = loss_shrink_maps

        return metrics


class DBLossV3(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta

        self.dice_loss = DiceLoss(eps=eps)
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, loss=self.dice_loss)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, preds, targets):

        metrics = {
            'loss_shrink_maps': 0,
            'loss_threshold_maps': 0,
            'loss_binary_maps': 0,
            'loss': 0,
        }

        for pred, batch, label in zip(preds, targets, ['even', 'odd']):
            shrink_maps = pred[:, 0, :, :]
            threshold_maps = pred[:, 1, :, :]

            loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
            loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])

            metrics['loss_shrink_maps'] += loss_shrink_maps
            metrics['loss_threshold_maps'] += loss_threshold_maps

            if pred.size()[1] > 2:
                binary_maps = pred[:, 2, :, :]
                loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
                metrics['loss_binary_maps'] += loss_binary_maps
                # alpha=1.0, beta=10
                loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
                metrics['loss'] += loss_all
                metrics[f"{label}_loss"] = loss_all
            else:
                metrics['loss'] += loss_shrink_maps
                metrics[f"{label}_loss"] = loss_shrink_maps

        return metrics