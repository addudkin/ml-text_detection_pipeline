import cv2
import pyclipper

from shapely.geometry import Polygon

import pandas as pd
import numpy as np
from terminaltables import AsciiTable
from utils.mean_precision_recall_fscore import mean_precision_recall_fscore_support
from typing import List, Union, Dict


class SegPostprocessing:
    def __init__(self, binarization_threshold=0.3, confidence_threshold=0.7, unclip_ratio=1.5, min_area=10.):
        self.binarization_threshold = binarization_threshold
        self.confidence_threshold = confidence_threshold
        self.unclip_ratio = unclip_ratio
        self.min_area = min_area

    def __call__(self, width, height, pred, return_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.size(0)):
            if return_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.binarization_threshold

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if score < self.confidence_threshold:
                continue

            if points.shape[0] > 2:
                curr_boxes = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(curr_boxes) == 0:
                    continue
                else:
                    for box in curr_boxes:
                        box = np.array(box).reshape(-1, 2)

                        if not isinstance(dest_width, int):
                            dest_width = dest_width.item()
                            dest_height = dest_height.item()

                        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
                        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
                        boxes.append(box)
                        scores.append(score)

        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

        num_contours = len(contours)
        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index].squeeze(1)

            score = self.box_score_fast(pred, contour)
            if score < self.confidence_threshold:
                continue

            points, sside = self.get_mini_boxes(contour)
            points = np.array(points)
            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)

            if len(box) < 1:
                continue

            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def unclip(self, box, unclip_ratio=1):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


class Evaluator(object):
    def __init__(self,):
        self.result = {
        'mean_precision': 0,
        'mean_recall': 0,
        'mean_fscore': 0,
        'mean_support': 0,
        'iou_thresholds': 0,
        'precisions': 0,
        'recalls': 0,
        'fscores': 0
    }

    def __call__(self, gt_polygons, pred_polygons):
        # mean precision recall fscore
        out = []
        for gt_p, pred_p in zip(gt_polygons, pred_polygons):
            out.append(mean_precision_recall_fscore_support(gt_p, pred_p))

        current_result = {
            'mean_precision': np.mean([o['mean_precision'] for o in out]),
            'mean_recall': np.mean([o['mean_recall'] for o in out]),
            'mean_fscore': np.mean([o['mean_fscore'] for o in out]),
            'mean_support': np.mean([o['support'] for o in out]),
            'iou_thresholds': out[0]['iou_thresholds'],
            'precisions': np.mean(np.stack([o['precisions'] for o in out]), axis=0),
            'recalls': np.mean(np.stack([o['recalls'] for o in out]), axis=0),
            'fscores': np.mean(np.stack([o['fscores'] for o in out]), axis=0)
        }
        for k, v in current_result.items():
            self.result[k]+=v

    def clear_data(self):
        for k, v in self.result.items():
            self.result[k] = 0


def show_metrics(
        metrics: Dict,
        epoch: Union[int, str] = None,
        batch_size: int = None
) -> None:
    """Print a table of Word Accuracy Rate (WAR) and Character Accuracy Rate (CAR) for each class in the given DataFrame.

    Args:
        metrics: A Pandas DataFrame containing WAR and CAR for each class.
        epoch: An optional epoch number to print at the beginning of the table.
    """
    metric_names = ['mean_precision', 'mean_recall', 'mean_fscore']
    if epoch is not None:
        print(f'Epoch number:{epoch}')
    ap_table = [["Metric name", "Value"]]
    for label in metric_names:
        print(label, metrics[label]/batch_size)
        #ap_table += [[label, f"{float(value):.5f}"]]
    #print(AsciiTable(ap_table).table)
    del ap_table


def draw_polygons(img, polygons, contours=False):
    img = img.copy()
    for p in polygons:
        p = np.array([p]).astype(np.int32)
        if contours:
            cv2.drawContours(img, p, -1, (0, 255, 0), 3)
        else:
            cv2.fillPoly(img, p, (0, 255, 0))
    return img
