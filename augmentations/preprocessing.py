import random
import cv2

import numpy as np

from shapely.geometry import Polygon

from typing import List


def pad_keypoint(keypoint, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right):
    x, y = keypoint
    x += w_pad_left
    y += h_pad_top
    return x, y


def pad_divisor(size, divisor):
    r = size // divisor
    if r == 0:
        return size
    else:
        return (r + 1) * divisor


def crop(img, polygons: List[Polygon], box: Polygon):
    h, w, c = img.shape

    points = box.boundary.xy
    x1, y1 = int(min(points[0])), int(min(points[1]))
    x3, y3 = int(max(points[0])), int(max(points[1]))

    cropped_img = img[y1:y3, x1:x3, :]

    cropped_polygons = []
    for p in polygons:
        inter = box.intersection(p)
        if inter.area > 0:
            cropped_polygons.append(inter)

    return cropped_img, cropped_polygons


def xywh2polygon(center_x: int, center_y: int, w: int, h: int):
    x1, y1 = center_x - w // 2, center_y - h // 2
    x3, y3 = center_x + w // 2, center_y + h // 2
    return np.array([
        [x1, y1],
        [x3, y1],
        [x3, y3],
        [x1, y3]
    ]).astype(np.int32)


def pad_if_needed(img: np.ndarray, polygons: List[np.ndarray],
                  min_height=640, min_width=640,
                  pad_height_divisor=None, pad_width_divisor=None,
                  border_mode=0, value=None, mode='center'):
    assert mode in ['center', 'right-bottom']
    h, w, c = img.shape
    target_h = h if h > min_height else min_height
    target_w = w if w > min_width else min_width
    if pad_height_divisor is not None:
        target_h = pad_divisor(target_h, pad_height_divisor)
    if pad_width_divisor is not None:
        target_w = pad_divisor(target_w, pad_width_divisor)

    if mode == 'right-bottom':
        h_pad_top = 0
        h_pad_bottom = target_h - h
        w_pad_left = 0
        w_pad_right = target_w - w
    elif mode == 'center':
        h_pad_top = (target_h - h) // 2
        h_pad_bottom = (target_h - h) // 2 + (target_h - h) % 2
        w_pad_left = (target_w - w) // 2
        w_pad_right = (target_w - w) // 2 + (target_w - w) % 2
    else:
        raise Exception('Not implemented mode: ', mode)

    padded_img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value=value)

    padded_polygons = []
    for polygon in polygons:
        points = [pad_keypoint(p, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right) for p in polygon]
        points = np.array(points)
        padded_polygons.append(points)

    return padded_img, padded_polygons


def pad_if_neededV2(
        img: np.ndarray, grouped_polygons: List[List],
        min_height=640, min_width=640,
        pad_height_divisor=None, pad_width_divisor=None,
        border_mode=0, value=None, mode='center'
):
    """
    Для 2х наборов полигонов
    """
    assert mode in ['center', 'right-bottom']
    h, w, c = img.shape
    target_h = h if h > min_height else min_height
    target_w = w if w > min_width else min_width
    if pad_height_divisor is not None:
        target_h = pad_divisor(target_h, pad_height_divisor)
    if pad_width_divisor is not None:
        target_w = pad_divisor(target_w, pad_width_divisor)

    if mode == 'right-bottom':
        h_pad_top = 0
        h_pad_bottom = target_h - h
        w_pad_left = 0
        w_pad_right = target_w - w
    elif mode == 'center':
        h_pad_top = (target_h - h) // 2
        h_pad_bottom = (target_h - h) // 2 + (target_h - h) % 2
        w_pad_left = (target_w - w) // 2
        w_pad_right = (target_w - w) // 2 + (target_w - w) % 2
    else:
        raise Exception('Not implemented mode: ', mode)

    padded_img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value=value)

    padded_polygons = []

    for polygons in grouped_polygons:
        gr_polys = []
        for polygon in polygons:
            points = [pad_keypoint(p, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right) for p in polygon]
            points = np.array(points)
            gr_polys.append(points)
        padded_polygons.append(gr_polys)

    return padded_img, padded_polygons


def center_crop(img: np.ndarray, polygons: List[np.ndarray], width: int, height: int, center_x: int = None,
                center_y: int = None):
    h, w, c = img.shape
    center_x = w // 2 if center_x is None else center_x
    center_y = h // 2 if center_y is None else center_y
    polygons = [Polygon(p.astype(np.int32)) for p in polygons]
    box = Polygon(xywh2polygon(center_x, center_y, width, height))
    cropped_img, cropped_polygons = crop(img, polygons, box)
    out_cropped_polygons = []
    for p in cropped_polygons:
        x, y = p.boundary.xy
        x = x - np.array(center_x - width // 2)
        y = y - np.array(center_y - height // 2)
        p = np.stack([x, y], axis=1).astype(np.int32)
        out_cropped_polygons.append(p)

    return cropped_img, out_cropped_polygons


def random_crop(img: np.ndarray, polygons: List[np.ndarray], width: int, height: int):
    h, w, c = img.shape

    random_x = random.randint(width // 2, w - width // 2)
    random_y = random.randint(height // 2, h - height // 2)
    return center_crop(img, polygons, width, height, random_x, random_y)


def resize_if_needed(img: np.ndarray, polygons: List[np.ndarray], max_size: int = 2048, return_ratio=False):
    h, w, c = img.shape
    max_side_size = max([h, w])

    if max_side_size < max_size:
        ratio = 1.
        if return_ratio:
            return img, polygons, ratio
        return img, polygons

    ratio = max_size / max_side_size

    target_h, target_w = int(h * ratio), int(w * ratio)

    resized_img = cv2.resize(img, (target_w, target_h))

    resized_polygons = []

    for polygon in polygons:
        r_p = [(int(x * ratio), int(y * ratio)) for x, y in polygon]
        resized_polygons.append(np.array(r_p))

    if return_ratio:
        return resized_img, resized_polygons, ratio

    return resized_img, resized_polygons


def resize_if_needed_even_odd(img: np.ndarray, polygons: List[List[np.ndarray]], max_size: int = 2048, return_ratio=False):
    h, w, c = img.shape
    max_side_size = max([h, w])

    if max_side_size < max_size:
        ratio = 1.
        if return_ratio:
            return img, polygons, ratio
        return img, polygons

    ratio = max_size / max_side_size

    target_h, target_w = int(h * ratio), int(w * ratio)

    resized_img = cv2.resize(img, (target_w, target_h))

    resized_polygons = []

    for group in polygons:
        group_polygons = []
        for line in group:
            line_polygons = []
            for polygon in line:
                r_p = [(int(x * ratio), int(y * ratio)) for x, y in polygon]
                line_polygons.append(np.array(r_p))
            group_polygons.append(line_polygons)
        resized_polygons.append(group_polygons)

    if return_ratio:
        return resized_img, resized_polygons, ratio

    return resized_img, resized_polygons