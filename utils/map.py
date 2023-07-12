import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import auc


# Функция для вычисления IoU между двуми полигонами
def calculate_iou(poly1, poly2):
    if poly1.intersects(poly2):
        iou = poly1.intersection(poly2).area / poly1.union(poly2).area
    else:
        iou = 0.0
    return iou


# Функция для вычисления mAP
def calculate_map(predict_np_array, target_np_array):
    # Преобразование np.array в списки полигонов
    predict_polygons = [Polygon(poly) for poly in predict_np_array]
    target_polygons = [Polygon(poly) for poly in target_np_array]

    # Уровни IoU
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    aps = []

    # Для каждого уровня IoU
    for iou_threshold in iou_thresholds:
        matches = []
        for target_poly in target_polygons:
            # Проверяем, есть ли хотя бы один предсказанный полигон с IoU больше порога
            match = any(calculate_iou(target_poly, predict_poly) >= iou_threshold for predict_poly in predict_polygons)
            matches.append(match)

        # Вычисляем precision и recall
        tp = sum(matches)  # True positives
        fp = len(predict_polygons) - tp  # False positives
        fn = len(target_polygons) - tp  # False negatives
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        # Строим Precision-Recall кривую и вычисляем площадь под ней
        precisions = [precision] + [1.0] * fn + [0.0] * fp
        recalls = [recall] + list(np.linspace(recall, 0.0, num=fn+fp))
        ap = auc(recalls, precisions)
        aps.append(ap)

    # Вычисляем среднее значение AP (mAP)
    mean_ap = np.mean(aps)

    return mean_ap

# Теперь вы можете вызвать функцию calculate_map с вашими списками
# mean_ap = calculate_map(predict_np_array, target_np_array)
# print(f"Mean Average Precision (mAP): {mean_ap}")
