"""
Утилиты для вычисления метрик качества детектирования (TPR, FDR).
"""

import numpy as np
from typing import List, Tuple, Dict
from .nms import compute_iou


def load_ground_truth(annotation_file: str, format: str = 'required', 
                     image_size: Tuple[int, int] = None) -> List[List[float]]:
    """
    Загрузка разметки (ground truth) из файла.
    
    Args:
        annotation_file: Путь к файлу с разметкой
        format: Формат разметки ('required', 'simple')
        image_size: Размер изображения (не используется, оставлен для совместимости)
    
    Returns:
        Список детекций в формате [x1, y1, x2, y2, class_id]
    """
    detections = []
    
    # Маппинг названий классов в COCO ID
    class_name_to_id = {
        'car': 2, 'CAR': 2, 'Car': 2,
        'bicycle': 1, 'BICYCLE': 1, 'Bicycle': 1,
        'motorcycle': 3, 'MOTORCYCLE': 3, 'Motorcycle': 3,
        'bus': 5, 'BUS': 5, 'Bus': 5,
        'truck': 7, 'TRUCK': 7, 'Truck': 7,
    }
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if format == 'required':
            # Формат: frame_id class_name x1 y1 x2 y2
            # 705 BUS 264 0 387 37
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    class_name = parts[1]
                    class_id = class_name_to_id.get(class_name, -1)
                    if class_id == -1:
                        # Попробуем uppercase/lowercase варианты
                        class_id = class_name_to_id.get(class_name.upper(), -1)
                        if class_id == -1:
                            class_id = class_name_to_id.get(class_name.lower(), -1)

                    if class_id == -1:
                        continue

                    try:
                        x1 = float(parts[2])
                        y1 = float(parts[3])
                        x2 = float(parts[4])
                        y2 = float(parts[5])

                        detections.append([x1, y1, x2, y2, class_id])
                    except ValueError:
                        continue

        elif format == 'simple':
            # Простой формат: x1 y1 x2 y2 class_id или x1 y1 x2 y2 class_name
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # Пробуем числовой формат
                        x1, y1, x2, y2, class_val = parts[:5]
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        
                        # Пробуем числовой ID
                        try:
                            class_id = int(float(class_val))
                        except (ValueError, TypeError):
                            # Если не число, ищем по названию
                            class_id = class_name_to_id.get(class_val, -1)
                            if class_id == -1:
                                continue
                        
                        detections.append([x1, y1, x2, y2, class_id])
                    except (ValueError, TypeError) as e:
                        continue  # Пропускаем некорректные строки
    
    except FileNotFoundError:
        print(f"Файл разметки не найден: {annotation_file}")
    except Exception as e:
        print(f"Ошибка при загрузке разметки: {e}")
    
    return detections


def load_ground_truth_from_dict(annotations: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
    """
    Загрузка разметки из словаря (ключ - имя файла, значение - список детекций).
    
    Args:
        annotations: Словарь с разметкой
        
    Returns:
        Тот же словарь (для совместимости)
    """
    return annotations


def calculate_metrics(predictions: List[List[float]], ground_truth: List[List[float]], 
                      iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Вычисление метрик TPR (True Positive Rate) и FDR (False Discovery Rate).
    
    Args:
        predictions: Список предсказаний в формате [x1, y1, x2, y2, class_id, confidence]
        ground_truth: Список истинных объектов в формате [x1, y1, x2, y2, class_id]
        iou_threshold: Порог IoU для совпадения детекций
        
    Returns:
        Кортеж (TPR, FDR)
    """
    if len(predictions) == 0 and len(ground_truth) == 0:
        return 1.0, 0.0
    
    if len(ground_truth) == 0:
        # Если нет истинных объектов, но есть предсказания - все это FP
        return 0.0, 1.0 if len(predictions) > 0 else 0.0
    
    if len(predictions) == 0:
        # Если нет предсказаний, но есть истинные объекты - все это FN
        return 0.0, 0.0
    
    # Преобразование в numpy массивы
    pred_boxes = np.array([p[:4] for p in predictions])
    pred_classes = np.array([int(p[4]) for p in predictions])
    pred_scores = np.array([p[5] if len(p) > 5 else 1.0 for p in predictions])
    
    gt_boxes = np.array([g[:4] for g in ground_truth])
    gt_classes = np.array([int(g[4]) for g in ground_truth])
    
    # Сортировка предсказаний по уверенности (от большей к меньшей)
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    # Отслеживание совпадений
    matched_gt = set()  # Индексы совпавших истинных объектов
    TP = 0  # True Positives
    FP = 0  # False Positives
    
    # Для каждого предсказания ищем лучшее совпадение
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_class = pred_classes[pred_idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # Ищем лучшее совпадение среди истинных объектов того же класса
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            
            if gt_classes[gt_idx] == pred_class:
                iou = compute_iou(pred_box, gt_box.reshape(1, -1))[0]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Если IoU выше порога - это True Positive
        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1
    
    # False Negatives - несовпавшие истинные объекты
    FN = len(ground_truth) - len(matched_gt)
    
    # Вычисление метрик
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FDR = FP / (FP + TP) if (FP + TP) > 0 else 0.0
    
    return TPR, FDR


def calculate_metrics_per_class(predictions: List[List[float]], 
                                ground_truth: List[List[float]],
                                iou_threshold: float = 0.5) -> Dict[int, Tuple[float, float]]:
    """
    Вычисление метрик отдельно для каждого класса.
    
    Args:
        predictions: Список предсказаний
        ground_truth: Список истинных объектов
        iou_threshold: Порог IoU
        
    Returns:
        Словарь {class_id: (TPR, FDR)}
    """
    # Группировка по классам
    all_classes = set()
    for p in predictions:
        if len(p) >= 5:
            all_classes.add(int(p[4]))
    for g in ground_truth:
        if len(g) >= 5:
            all_classes.add(int(g[4]))
    
    metrics_per_class = {}
    for class_id in all_classes:
        # Фильтрация по классу
        pred_class = [p for p in predictions if len(p) >= 5 and int(p[4]) == class_id]
        gt_class = [g for g in ground_truth if len(g) >= 5 and int(g[4]) == class_id]
        
        TPR, FDR = calculate_metrics(pred_class, gt_class, iou_threshold)
        metrics_per_class[class_id] = (TPR, FDR)
    
    return metrics_per_class

