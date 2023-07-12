import cv2
import pyclipper

import numpy as np
import torch.nn as nn

from PIL import Image
from typing import List, Union, Dict, Tuple
from utils.types import SegmentationMaskResult, FieldMask, RotationCls


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def roll_top_left_first(coords: np.array) -> np.array:
    for _ in range(4):
        distances = np.linalg.norm(coords, 2, axis=1)
        is_first = np.argmin(distances) == 0

        if is_first:
            break
        coords = np.roll(coords, 1, 0)
    else:
        raise Exception("Failed to find correct sort")
    return coords


def rotate_bbox(bbox: np.ndarray,
                rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Rotate bbox using rotation_matrix

    Args:
      bbox: bbox in Quadrangle type coords format
      rotation_matrix: opencv rotation matrix
    """
    bbox = np.array(bbox).reshape((-1, 1, 2))
    bbox = cv2.transform(bbox, rotation_matrix)
    return np.squeeze(bbox)


def align_coords(boxes: List[np.ndarray]) -> np.ndarray:
    """
    Estimate mean rotation angle(MRA) for all bboxes. MRA is a mean of rotation
    angles weighted by individual bbox area when taking mean.

    Args:
      boxes: list of boxes in Quadrangle type coords format
    Return:
       aligned_boxes:  array of all boxes rotated by MRA
    """
    aligned_boxes = np.zeros(shape=(len(boxes), 4, 2), dtype=np.int32)
    all_theta = []
    all_area = []
    for point in boxes:
        _, (w, h), theta = cv2.minAreaRect(point)
        if theta >= 45:
            theta -= 90
        all_theta.append(theta)
        all_area.append(w * h)
    theta = np.average(all_theta,
                       weights=all_area)
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), theta, 1.0)
    for i, box in enumerate(boxes):
        aligned_boxes[i] = rotate_bbox(box, rotation_matrix)
    return aligned_boxes


def sort_bboxes(boxes: List[np.ndarray],
                intersection_threshold: float = 0.5) -> List[List[int]]:
    """
    Sort word bboxes in natural reading order

    Args:
        boxes: list of boxes in Quadrangle type coords format
        intersection_threshold: bboxes intersection threshold on vertical axis to assume they are on the same line
    Notes:
        intersection_threshold values closer to 0 work better for cases of rotated bboxes and if text
        jumps all over the line.
        Values closer to 1 work better if bboxes in one line are aligned and have no rotation.
    Returns:
        sorted_boxes: list of sorted word boxes
        sorted_ids: list of index boxes after sorting

    """

    if not boxes:
        return [[]]

    if len(boxes) == 1:
        return [[0]]

    boxes = np.array(boxes)
    indices = np.arange(len(boxes))  # [ 0,1,2, ...]
    aligned_boxes = align_coords(boxes)
    # sort by center y, height of normalized boxes
    # cv2.minAreaRect returns ((cx,cy),(w,h),theta)
    indices = np.asarray(sorted(indices, key=lambda i: (cv2.minAreaRect(aligned_boxes[i])[0][1],
                                                        cv2.minAreaRect(aligned_boxes[i])[1][1])))

    y1 = aligned_boxes[:, :, 1].min(axis=1)  # top left
    y2 = aligned_boxes[:, :, 1].max(axis=1)  # bottom right
    heights = y2 - y1

    sorted_boxes = []
    sorted_ids = []

    while indices.size:

        idx = indices[0]
        buffer_merge_ids = [idx]
        indices = indices[1:]
        # ndarray of shape : (len(indices),)
        max_y1 = np.maximum(y1[idx], y1[indices])
        min_y2 = np.minimum(y2[idx], y2[indices])
        intersection_height = np.maximum(0.0, min_y2 - max_y1)
        overlap = intersection_height / heights[idx]

        for idx in indices[overlap >= intersection_threshold]:
            buffer_merge_ids.append(idx)
        # sort boxes that are on the same line in the left corner
        buffer_merge_ids = sorted(buffer_merge_ids, key=lambda i: boxes[i][0][0])
        sorted_ids.append(buffer_merge_ids)
        sorted_boxes.extend(boxes[buffer_merge_ids])
        # filter from overlap boxes
        indices = indices[overlap < intersection_threshold]

    return sorted_ids


class FullTextPostProcessor:
    def __init__(self, shrink_ratio, intersection_threshold, confidence_threshold, binarization_threshold, min_area):
        self.shrink_ratio = shrink_ratio
        self.intersection_threshold = intersection_threshold
        self.confidence_threshold = confidence_threshold
        self.binarization_threshold = binarization_threshold
        self.min_area = min_area

    def find_word_boxes(self,
                        mask: np.ndarray,
                        scale_factors: Tuple[float, float]):

        boxes = []
        confidences = []
        heights = []

        height, width = mask.shape[:2]
        scale = 1 * 10e-6 * height * width

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for label_id in range(n_labels):
            x1 = stats[label_id, cv2.CC_STAT_LEFT]
            y1 = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]
            x2, y2 = x1 + w, y1 + h
            confidence = mask[y1:y2 + 1, x1:x2 + 1].mean() / 255.

            if area < self.min_area:
                continue

            if confidence < self.confidence_threshold:
                continue

            if w * h < scale:
                continue

            contours, _ = cv2.findContours((labels[y1:y2 + 1, x1:x2 + 1] == label_id).astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            epsilon = 0.001 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            contours = self.extend_box(approx)

            if len(contours) < 4:
                continue

            oriented_rect = cv2.minAreaRect(contours)

            w, h = oriented_rect[1]

            rect = cv2.boxPoints(oriented_rect)
            rect[:, 0] += x1
            rect[:, 1] += y1
            rect = roll_top_left_first(rect)
            rect[:, 0] *= scale_factors[0]
            rect[:, 1] *= scale_factors[1]
            box = rect.round().astype('int32')
            boxes.append(box)
            confidences.append(confidence)

            if abs(oriented_rect[-1]) >= 45:
                w, h = h, w
            heights.append(h)

        boxes = np.array(boxes)
        confidences = np.array(confidences)
        heights = np.array(heights)
        median_h = np.median(heights)
        boxes = boxes[heights < 2. * median_h]
        confidences = confidences[heights < 2. * median_h]

        return boxes.tolist(), confidences.tolist()

    def get_distance(self,
                     polygon: np.ndarray, ) -> float:
        """
        """
        area = cv2.contourArea(polygon)
        perimeter = cv2.arcLength(polygon, True)
        distance = area * self.shrink_ratio / perimeter
        return distance

    def extend_box(self,
                   box: np.ndarray) -> np.ndarray:
        subject = [tuple(p[0]) for p in box]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject,
                        pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        extend_polygon = padding.Execute(self.get_distance(box))
        if extend_polygon:
            extend_polygon = extend_polygon[0]
        return np.array(extend_polygon)

    def run(self,
            args: SegmentationMaskResult
            ) -> List[FieldMask]:

        x1, y1, x2, y2 = args.coords

        prediction_labels = args.prediction_labels[y1:y2 + 1, x1:x2 + 1]
        prediction_labels = prediction_labels.detach().cpu().numpy() > self.binarization_threshold
        prediction_labels = prediction_labels.astype(np.uint8)
        boxes, confidences = self.find_word_boxes(prediction_labels, args.scale)

        sorted_boxes = []
        sorted_ids = sort_bboxes(boxes, self.intersection_threshold)
        for idx in sum(sorted_ids, []):
            box = boxes[idx]
            confidence = confidences[idx]
            sorted_boxes.append(FieldMask(
                coords=(tuple(box[0]),
                        tuple(box[1]),
                        tuple(box[2]),
                        tuple(box[3])),
                confidence=confidence,
                rotation_cls=RotationCls.deg0)
            )

        return sorted_boxes


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


def draw_polygons(img: np.ndarray, polygons: List[np.ndarray]):
    img = img.copy()
    for p in polygons:
        p = np.array([p]).astype(np.int32)
        cv2.drawContours(img, p, -1, (0, 255, 0), 3)
    return Image.fromarray(img)
#
# def draw_polygons(img, polygons, contours=False):
#     img = img.copy()
#     for p in polygons:
#         p = np.array([p]).astype(np.int32)
#         if contours:
#             cv2.drawContours(img, p, -1, (0, 255, 0), 3)
#         else:
#             cv2.fillPoly(img, p, (0, 255, 0))
#     return Image.fromarray(img)
