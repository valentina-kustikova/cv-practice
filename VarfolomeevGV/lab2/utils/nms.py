"""
Утилиты для Non-Maximum Suppression и вычисления IoU.
"""

import numpy as np
from typing import List, Tuple


def compute_iou(box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Вычисление Intersection over Union (IoU) между одним боксом и массивом боксов.
    
    Args:
        box1: Один бокс в формате [x1, y1, x2, y2]
        boxes: Массив боксов в формате [N, 4] где каждая строка [x1, y1, x2, y2]
        
    Returns:
        Массив IoU значений [N]
    """
    # Извлечение координат
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2 = boxes[:, 0]
    y1_2 = boxes[:, 1]
    x2_2 = boxes[:, 2]
    y2_2 = boxes[:, 3]
    
    # Вычисление координат пересечения
    x_left = np.maximum(x1_1, x1_2)
    y_top = np.maximum(y1_1, y1_2)
    x_right = np.minimum(x2_1, x2_2)
    y_bottom = np.minimum(y2_1, y2_2)
    
    # Площадь пересечения
    width_inter = np.maximum(0, x_right - x_left)
    height_inter = np.maximum(0, y_bottom - y_top)
    intersection_area = width_inter * height_inter
    
    # Площади боксов (проверка на валидность)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Проверка на валидные площади (должны быть положительными)
    area1 = np.maximum(0, area1)
    area2 = np.maximum(0, area2)
    
    # Площадь объединения
    union_area = area1 + area2 - intersection_area
    
    # Избегаем деления на ноль и NaN
    # Используем np.divide с where для безопасного деления
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.divide(intersection_area, union_area, 
                       out=np.zeros_like(intersection_area, dtype=float), 
                       where=(union_area > 1e-8))
    
    # Убираем NaN значения
    iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    
    return iou


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, 
                       iou_threshold: float = 0.4) -> np.ndarray:
    """
    Non-Maximum Suppression для удаления дублирующихся детекций.
    
    Args:
        boxes: Массив боксов в формате [N, 4] где каждая строка [x1, y1, x2, y2]
        scores: Массив уверенностей [N]
        iou_threshold: Порог IoU для удаления дубликатов
        
    Returns:
        Индексы боксов, которые нужно оставить
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Сортировка по убыванию уверенности
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        # Берем бокс с максимальной уверенностью
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Вычисляем IoU с остальными боксами
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = compute_iou(current_box, other_boxes)
        
        # Удаляем боксы с высоким IoU (дубликаты)
        mask = ious < iou_threshold
        indices = indices[1:][mask]
    
    return np.array(keep, dtype=np.int32)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                  iou_threshold: float = 0.4) -> np.ndarray:
    """
    Non-Maximum Suppression, применяемый отдельно для каждого класса.
    
    Args:
        boxes: Массив боксов [N, 4]
        scores: Массив уверенностей [N]
        class_ids: Массив идентификаторов классов [N]
        iou_threshold: Порог IoU
        
    Returns:
        Индексы боксов, которые нужно оставить
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    unique_classes = np.unique(class_ids)
    keep_all = []
    
    for cls_id in unique_classes:
        # Маска для текущего класса
        mask = class_ids == cls_id
        if not np.any(mask):
            continue
        
        # Индексы элементов текущего класса
        class_indices = np.where(mask)[0]
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        
        # NMS для текущего класса
        keep_class = non_max_suppression(class_boxes, class_scores, iou_threshold)
        
        # Сохраняем исходные индексы
        keep_all.extend(class_indices[keep_class])
    
    return np.array(keep_all, dtype=np.int32)

