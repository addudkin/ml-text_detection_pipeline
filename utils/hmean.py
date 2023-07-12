from typing import List, Optional, Union, Tuple

import numpy as np
import pyclipper
from shapely.geometry import Polygon, MultiPolygon


def polys2shapely(polygons: List[np.ndarray]) -> List[Polygon]:
    """Convert a nested list of boundaries to a list of Polygons.

    Args:
        polygons (list): The point coordinates of the instance boundary.

    Returns:
        list: Converted shapely.Polygon.
    """
    return [poly2shapely(polygon) for polygon in polygons]


def poly_make_valid(poly: Polygon) -> Polygon:
    """Convert a potentially invalid polygon to a valid one by eliminating
    self-crossing or self-touching parts. Note that if the input is a line, the
    returned polygon could be an empty one.

    Args:
        poly (Polygon): A polygon needed to be converted.

    Returns:
        Polygon: A valid polygon, which might be empty.
    """
    assert isinstance(poly, Polygon)
    fixed_poly = poly if poly.is_valid else poly.buffer(0)
    # Sometimes the fixed_poly is still a MultiPolygon,
    # so we need to find the convex hull of the MultiPolygon, which should
    # always be a Polygon (but could be empty).
    if not isinstance(fixed_poly, Polygon):
        fixed_poly = fixed_poly.convex_hull
    return fixed_poly


def poly_intersection(poly_a: Polygon,
                      poly_b: Polygon,
                      invalid_ret: Optional[Union[float, int]] = None,
                      return_poly: bool = False
                      ) -> Tuple[float, Optional[Polygon]]:
    """Calculate the intersection area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to None.
        return_poly (bool): Whether to return the polygon of the intersection
            Defaults to False.

    Returns:
        float or tuple(float, Polygon): Returns the intersection area or
        a tuple ``(area, Optional[poly_obj])``, where the `area` is the
        intersection area between two polygons and `poly_obj` is The Polygon
        object of the intersection area. Set as `None` if the input is invalid.
        Set as `None` if the input is invalid. `poly_obj` will be returned
        only if `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.intersection(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_union(
    poly_a: Polygon,
    poly_b: Polygon,
    invalid_ret: Optional[Union[float, int]] = None,
    return_poly: bool = False
) -> Tuple[float, Optional[Union[Polygon, MultiPolygon]]]:
    """Calculate the union area between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        invalid_ret (float or int, optional): The return value when the
            invalid polygon exists. If it is not specified, the function
            allows the computation to proceed with invalid polygons by
            cleaning the their self-touching or self-crossing parts.
            Defaults to False.
        return_poly (bool): Whether to return the polygon of the union.
            Defaults to False.

    Returns:
        tuple: Returns a tuple ``(area, Optional[poly_obj])``, where
        the `area` is the union between two polygons and `poly_obj` is the
        Polygon or MultiPolygon object of the union of the inputs. The type
        of object depends on whether they intersect or not. Set as `None`
        if the input is invalid. `poly_obj` will be returned only if
        `return_poly` is `True`.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    assert invalid_ret is None or isinstance(invalid_ret, (float, int))

    if invalid_ret is None:
        poly_a = poly_make_valid(poly_a)
        poly_b = poly_make_valid(poly_b)

    poly_obj = None
    area = invalid_ret
    if poly_a.is_valid and poly_b.is_valid:
        poly_obj = poly_a.union(poly_b)
        area = poly_obj.area
    return (area, poly_obj) if return_poly else area


def poly_iou(poly_a: Polygon,
             poly_b: Polygon,
             zero_division: float = 0.) -> float:
    """Calculate the IOU between two polygons.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        zero_division (float): The return value when invalid polygon exists.

    Returns:
        float: The IoU between two polygons.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)
    area_inters = poly_intersection(poly_a, poly_b)
    area_union = poly_union(poly_a, poly_b)
    return area_inters / area_union if area_union != 0 else zero_division


def is_poly_inside_rect(poly: np.ndarray, rect: np.ndarray) -> bool:
    """Check if the polygon is inside the target region.
        Args:
            poly (ArrayLike): Polygon in shape (N, ).
            rect (ndarray): Target region [x1, y1, x2, y2].

        Returns:
            bool: Whether the polygon is inside the cropping region.
        """

    poly = poly2shapely(poly)
    rect = poly2shapely(bbox2poly(rect))
    return rect.contains(poly)


def bbox2poly(bbox: np.ndarray, mode: str = 'xyxy') -> np.array:
    """Converting a bounding box to a polygon.

    Args:
        bbox (ArrayLike): A bbox. In any form can be accessed by 1-D indices.
         E.g. list[float], np.ndarray, or torch.Tensor. bbox is written in
            [x1, y1, x2, y2].
        mode (str): Specify the format of bbox. Can be 'xyxy' or 'xywh'.
            Defaults to 'xyxy'.

    Returns:
        np.array: The converted polygon [x1, y1, x2, y1, x2, y2, x1, y2].
    """
    assert len(bbox) == 4
    if mode == 'xyxy':
        x1, y1, x2, y2 = bbox
        poly = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    elif mode == 'xywh':
        x, y, w, h = bbox
        poly = np.array([x, y, x + w, y, x + w, y + h, x, y + h])
    else:
        raise NotImplementedError('Not supported mode.')

    return poly


def poly2shapely(polygon: np.ndarray) -> Polygon:
    """Convert a polygon to shapely.geometry.Polygon.

    Args:
        polygon (ArrayLike): A set of points of 2k shape.

    Returns:
        polygon (Polygon): A polygon object.
    """
    polygon = np.array(polygon, dtype=np.float32)
    assert polygon.size % 2 == 0 and polygon.size >= 6

    polygon = polygon.reshape([-1, 2])
    return Polygon(polygon)


def offset_polygon(poly: np.ndarray, distance: float) -> np.ndarray:
    """Offset (expand/shrink) the polygon by the target distance. It's a
    wrapper around pyclipper based on Vatti clipping algorithm.

    Warning:
        Polygon coordinates will be casted to int type in PyClipper. Mind the
        potential precision loss caused by the casting.

    Args:
        poly (ArrayLike): A polygon. In any form can be converted
            to an 1-D numpy array. E.g. list[float], np.ndarray,
            or torch.Tensor. Polygon is written in
            [x1, y1, x2, y2, ...].
        distance (float): The offset distance. Positive value means expanding,
            negative value means shrinking.

    Returns:
        np.array: 1-D Offsetted polygon ndarray in float32 type. If the
        result polygon is invalid or has been split into several parts,
        return an empty array.
    """
    poly = np.array(poly).reshape(-1, 2)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    # Returned result will be in type of int32, convert it back to float32
    # following MMOCR's convention
    result = np.array(pco.Execute(distance))
    if len(result) > 0 and isinstance(result[0], list):
        # The processed polygon has been split into several parts
        result = np.array([])
    result = result.astype(np.float32)
    # Always use the first polygon since only one polygon is expected
    # But when the resulting polygon is invalid, return the empty array
    # as it is
    return result if len(result) == 0 else result[0].flatten()


def compute_hmean(accum_hit_recall, accum_hit_prec, gt_num, pred_num):
    # TODO Add typehints
    """Compute hmean given hit number, ground truth number and prediction
    number.

    Args:
        accum_hit_recall (int|float): Accumulated hits for computing recall.
        accum_hit_prec (int|float): Accumulated hits for computing precision.
        gt_num (int): Ground truth number.
        pred_num (int): Prediction number.

    Returns:
        recall (float):  The recall value.
        precision (float): The precision value.
        hmean (float): The hmean value.
    """

    assert isinstance(accum_hit_recall, (float, int))
    assert isinstance(accum_hit_prec, (float, int))

    assert isinstance(gt_num, int)
    assert isinstance(pred_num, int)
    assert accum_hit_recall >= 0.0
    assert accum_hit_prec >= 0.0
    assert gt_num >= 0.0
    assert pred_num >= 0.0

    if gt_num == 0:
        recall = 1.0
        precision = 0.0 if pred_num > 0 else 1.0
    else:
        recall = float(accum_hit_recall) / gt_num
        precision = 0.0 if pred_num == 0 else float(accum_hit_prec) / pred_num

    denom = recall + precision

    hmean = 0.0 if denom == 0 else (2.0 * precision * recall / denom)

    return recall, precision, hmean


def poly_iou_optimized(poly_a: Polygon, poly_b: Polygon, zero_division: float = 0.) -> float:
    """Calculate the IOU between two polygons with bounding box check for efficiency.

    Args:
        poly_a (Polygon): Polygon a.
        poly_b (Polygon): Polygon b.
        zero_division (float): The return value when invalid polygon exists.

    Returns:
        float: The IoU between two polygons.
    """
    assert isinstance(poly_a, Polygon)
    assert isinstance(poly_b, Polygon)

    # Obtain bounding box (minx, miny, maxx, maxy)
    box_a = poly_a.bounds
    box_b = poly_b.bounds

    # Check if bounding boxes intersect, if not, return 0 immediately
    if (box_a[2] < box_b[0] or box_a[0] > box_b[2] or
            box_a[3] < box_b[1] or box_a[1] > box_b[3]):
        return 0.0

    area_inters = poly_intersection(poly_a, poly_b)
    area_union = poly_union(poly_a, poly_b)
    return area_inters / area_union if area_union != 0 else zero_division


def HmeanIOUMetric(pred_polys_list: List[np.array], gt_polys_list: List[np.array],
                   match_iou_thr: float = 0.5,
                   strategy: str = 'vanilla') -> dict:

    assert strategy in ['max_matching', 'vanilla']
    best_eval_results = dict(hmean=-1)

    dataset_gt_num = 0

    gt_polys = polys2shapely(gt_polys_list)
    pred_polys = polys2shapely(pred_polys_list)

    gt_num = len(gt_polys)
    pred_num = len(pred_polys)
    iou_metric = np.zeros([gt_num, pred_num])

    for pred_mat_id, pred_poly in enumerate(pred_polys):
        for gt_mat_id, gt_poly in enumerate(gt_polys):
            iou_metric[gt_mat_id, pred_mat_id] = poly_iou_optimized(gt_poly, pred_poly)#poly_iou(gt_poly, pred_poly)

    dataset_gt_num += iou_metric.shape[0]

    matched_metric = iou_metric > match_iou_thr
    matched_gt_indexes = set()
    matched_pred_indexes = set()
    for gt_idx, pred_idx in zip(*np.nonzero(matched_metric)):
        if gt_idx in matched_gt_indexes or pred_idx in matched_pred_indexes:
            continue
        matched_gt_indexes.add(gt_idx)
        matched_pred_indexes.add(pred_idx)

    dataset_hit_num = len(matched_gt_indexes)
    dataset_pred_num = np.sum(matched_metric)

    recall, precision, hmean = compute_hmean(int(dataset_hit_num), int(dataset_hit_num), int(dataset_gt_num),
                                             int(dataset_pred_num))

    eval_results = dict(precision=precision, recall=recall, hmean=hmean)
    if eval_results['hmean'] > best_eval_results['hmean']:
        best_eval_results = eval_results

    return best_eval_results



# def HmeanIOUMetric(pred_polys_list: List[np.array], gt_polys_list: List[np.array],
#                    match_iou_thr: float = 0.5,
#                    ignore_precision_thr: float = 0.5,
#                    pred_score_thrs: List[float] = list(np.arange(start=0.3, stop=0.9, step=0.1)),
#                    strategy: str = 'vanilla') -> dict:
#     assert strategy in ['max_matching', 'vanilla']
#     best_eval_results = dict(hmean=-1)
#
#     dataset_pred_num = np.zeros_like(pred_score_thrs)
#     dataset_hit_num = np.zeros_like(pred_score_thrs)
#     dataset_gt_num = 0
#
#     gt_polys = polys2shapely(gt_polys_list)
#     pred_polys = polys2shapely(pred_polys_list)
#
#     gt_num = len(gt_polys)
#     pred_num = len(pred_polys)
#     iou_metric = np.zeros([gt_num, pred_num])
#
#     for pred_mat_id, pred_poly in enumerate(pred_polys):
#         for gt_mat_id, gt_poly in enumerate(gt_polys):
#             iou_metric[gt_mat_id, pred_mat_id] = poly_iou(gt_poly, pred_poly)
#
#     dataset_gt_num += iou_metric.shape[0]
#
#     for i, pred_score_thr in enumerate(pred_score_thrs):
#         matched_metric = iou_metric > match_iou_thr
#         if strategy == 'max_matching':
#             csr_matched_metric = csr_matrix(matched_metric)
#             matched_preds = maximum_bipartite_matching(csr_matched_metric, perm_type='row')
#             dataset_hit_num[i] += np.sum(matched_preds != -1)
#         else:
#             matched_gt_indexes = set()
#             matched_pred_indexes = set()
#             for gt_idx, pred_idx in zip(*np.nonzero(matched_metric)):
#                 if gt_idx in matched_gt_indexes or pred_idx in matched_pred_indexes:
#                     continue
#                 matched_gt_indexes.add(gt_idx)
#                 matched_pred_indexes.add(pred_idx)
#             dataset_hit_num[i] += len(matched_gt_indexes)
#         dataset_pred_num[i] += np.sum(matched_metric)
#
#     for i, pred_score_thr in enumerate(pred_score_thrs):
#         recall, precision, hmean = compute_hmean(int(dataset_hit_num[i]), int(dataset_hit_num[i]), int(dataset_gt_num),
#                                                  int(dataset_pred_num[i]))
#         eval_results = dict(precision=precision, recall=recall, hmean=hmean)
#         if eval_results['hmean'] > best_eval_results['hmean']:
#             best_eval_results = eval_results
#
#     return best_eval_results