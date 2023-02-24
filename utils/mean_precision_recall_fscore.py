import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from sklearn.metrics import precision_recall_fscore_support


def mean_precision_recall_fscore_support(gt_points, pred_points, iou_thresholds=None):
    '''
    gt_points: N:K:2 - polygons of one example
    pred_points: N:K:2 - polygons of one example
    iou_thresholds: Default [0.3, 0.95, 0.05)
    '''
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.3, 0.95, 0.05)
    gt_polygons = [points2polygon(p, idx) for idx, p in enumerate(gt_points)]
    pred_polygons = [points2polygon(p, idx) for idx, p in enumerate(pred_points)]
    s = STRtree(gt_polygons)
    iou_matrix = np.full((len(gt_polygons), len(pred_polygons)), fill_value=0, dtype=float)

    for pred_idx, pred_polygon in enumerate(pred_polygons):
        result = s.query(pred_polygon)
        if len(result) > 0:
            for gt_idx, gt_polygon in enumerate(result):
                iou_matrix[gt_idx, pred_idx] = get_iou(pred_polygon, gt_polygon)

    ps, rs, fs, ss, ts = [], [], [], [], []
    for t in iou_thresholds:
        t = round(t, 2)
        p, r, f, s = precision_recall_fscore_support(iou_matrix, iou_threshold=t)
        ps.append(p)
        rs.append(r)
        fs.append(f)
        ss.append(s)
        ts.append(t)

    return {
        'mean_precision': np.mean(ps),
        'mean_recall': np.mean(rs),
        'mean_fscore': np.mean(fs),
        'support': np.mean(ss),
        'iou_thresholds': iou_thresholds,
        'precisions': ps,
        'recalls': rs,
        'fscores': fs
    }


def points2polygon(points, idx) -> Polygon:
    points = np.array(points).flatten()

    point_x = points[0::2]
    point_y = points[1::2]

    if len(point_x) < 3 or len(point_y) < 3:
        raise Exception('Implement approximation to 4 points as minimum')

    pol = Polygon(np.stack([point_x, point_y], axis=1)).buffer(0)

    return pol


def get_iou(p1: Polygon, p2: Polygon) -> float:
    intersection = p1.intersection(p2).area
    union = p1.area + p2.area
    return intersection / (union - intersection)


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def associate_gt_to_preds(iou_matrix, iou_threshold=0.5):
    gt_num, pred_num = iou_matrix.shape
    matched_indices = linear_assignment(-iou_matrix)
    matched_ious = np.array([iou_matrix[i, j] for i, j in matched_indices])
    tp = sum(matched_ious > iou_threshold)
    fp = pred_num - tp
    fn = gt_num - tp
    return tp, fp, fn


def precision_recall_fscore_support(iou_matrix, iou_threshold=0.5):
    tp, fp, fn = associate_gt_to_preds(iou_matrix, iou_threshold)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.
    recall = tp / (tp + fn) if tp + fn > 0 else 0.
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
    support = tp + fp + fn

    return precision, recall, fscore, support
