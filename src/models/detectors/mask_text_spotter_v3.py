from omegaconf import DictConfig
from torch import nn
from torchvision.ops import MultiScaleRoIAlign

from src.losses import build_loss
from src.models.detectors.db2 import DB
from src.models.heads2.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from src.models.heads2.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from src.models.heads2.roi_heads import RoIHeads
from src.postprocessing.seg_postrocessing_proposals import SEGPostProcessor


class MaskSpotter(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        self.DB = DB(cfg.model.DB)
        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'],
                                               output_size=7,
                                               sampling_ratio=2)
        resolution = self.box_roi_pool.output_size[0]
        representation_size = 1024
        out_channels = cfg.model.DB.in_channels
        self.post_processing = SEGPostProcessor(top_n=2000,
                                                binary_thresh=0.1,
                                                box_thresh=0.1,
                                                min_size=5,
                                                expand_ratio=2.25)
        self.db_loss = build_loss(cfg.loss.type, **cfg.loss.args)

        self.box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        representation_size = 1024
        self.box_predictor = FastRCNNPredictor(
            representation_size,
            2)

        self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        self.mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256
        mask_dim_reduced = 256
        self.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                                mask_dim_reduced, 2)

        self.roi_heads = RoIHeads(
            self.box_roi_pool,
            self.box_head,
            self.box_predictor,
            fg_iou_thresh=cfg.model.roi_head.args.fg_iou_thresh,
            bg_iou_thresh=cfg.model.roi_head.args.bg_iou_thresh,
            batch_size_per_image=cfg.model.roi_head.args.batch_size_per_image,
            positive_fraction=cfg.model.roi_head.args.positive_fraction,
            mask_roi_pool=self.mask_roi_pool,
            mask_head=self.mask_head,
            mask_predictor=self.mask_predictor,
            detections_per_img=cfg.model.roi_head.args.detections_per_img,
            score_thresh=cfg.model.roi_head.args.score_thresh,
            nms_thresh=cfg.model.roi_head.args.nms_thresh,
        )

    def forward(self, images, targets, shrink_maps, shrink_masks, threshold_maps, threshold_masks):
        db_preds, features = self.DB(images)
        proposals, boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch = self.post_processing(
            db_preds[:, [2], :, :])

        losses = self.db_loss(db_preds, {
            'shrink_map': shrink_maps,
            'shrink_mask': shrink_masks,
            'threshold_map': threshold_maps,
            'threshold_mask': threshold_masks
        })
        b, c, h, w = images.shape
        image_size = b * [(h, w)]
        model_result, detector_losses = self.roi_heads(features, proposals, image_size, targets)

        detector_losses["db_loss"] = losses["loss"]

        return model_result, detector_losses, polygons_batch
