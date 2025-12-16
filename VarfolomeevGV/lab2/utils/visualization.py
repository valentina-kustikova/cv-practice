"""
Утилиты для визуализации результатов детектирования.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def generate_colors(num_classes: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Генерация уникальных цветов для каждого класса.
    
    Args:
        num_classes: Количество классов
        seed: Seed для воспроизводимости
        
    Returns:
        Список цветов в формате BGR (для OpenCV)
    """
    np.random.seed(seed)
    colors = []
    
    for i in range(num_classes):
        # Генерируем яркие, различимые цвета
        hue = int(180 * i / num_classes)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color_bgr)))
    
    return colors


def draw_detections(image: np.ndarray, detections: List[List[float]], 
                   class_names: List[str], colors: Optional[List[Tuple[int, int, int]]] = None,
                   show_class_name: bool = True, show_confidence: bool = True,
                   font_scale: float = 0.5, thickness: int = 2) -> np.ndarray:
    """
    Отрисовка детекций на изображении.
    
    Требования:
    - Прямоугольники разных цветов для разных классов
    - В левом верхнем углу каждого прямоугольника: название класса и confidence (3 знака после запятой)
    - Над прямоугольником: название класса объекта
    
    Args:
        image: Входное изображение в формате BGR
        detections: Список детекций в формате [x1, y1, x2, y2, class_id, confidence]
        class_names: Список названий классов
        colors: Список цветов для классов (если None, генерируются автоматически)
        show_class_name: Показывать название класса
        show_confidence: Показывать уверенность
        font_scale: Масштаб шрифта
        thickness: Толщина линий
        
    Returns:
        Изображение с нарисованными детекциями
    """
    result_image = image.copy()
    
    if len(detections) == 0:
        return result_image
    
    # Генерация цветов, если не предоставлены
    if colors is None:
        max_class_id = max(int(d[4]) for d in detections if len(d) >= 5)
        colors = generate_colors(max_class_id + 1)
    
    # Отрисовка каждой детекции
    h, w = image.shape[:2]
    
    for det in detections:
        if len(det) < 5:
            continue
        
        x1, y1, x2, y2 = det[:4]
        class_id = int(det[4])
        confidence = det[5] if len(det) > 5 else 1.0
        
        # Проверка и исправление координат
        # Убеждаемся, что x1 < x2 и y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Ограничение координат границами изображения
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Пропускаем некорректные боксы (слишком маленькие или вырожденные)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        
        # Получение цвета для класса
        color = colors[class_id % len(colors)] if class_id < len(colors) else (0, 255, 0)
        
        # Отрисовка прямоугольника
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Подготовка текста
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        
        text_parts = []
        if show_class_name:
            text_parts.append(class_name)
        if show_confidence:
            text_parts.append(f"{confidence:.3f}")
        
        text = ": ".join(text_parts) if len(text_parts) > 1 else text_parts[0] if text_parts else ""
        
        # Вычисление размера текста
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Фон для текста (для читаемости)
        text_y = y1 - 10 if y1 > 20 else y1 + text_height + 10
        cv2.rectangle(
            result_image,
            (x1, text_y - text_height - baseline),
            (x1 + text_width, text_y + baseline),
            color,
            -1
        )
        
        # Отрисовка текста
        cv2.putText(
            result_image,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # Белый текст
            thickness,
            cv2.LINE_AA
        )
        
        # Надпись над прямоугольником (название класса)
        if show_class_name:
            label_y = max(y1 - 5, 15)
            cv2.putText(
                result_image,
                class_name,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.8,
                color,
                thickness,
                cv2.LINE_AA
            )
    
    return result_image


def draw_metrics(image: np.ndarray, tpr: float, fdr: float,
                 font_scale: float = 0.7, thickness: int = 2,
                 position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Отрисовка метрик TPR и FDR в верхнем левом углу изображения.
    
    Args:
        image: Входное изображение в формате BGR
        tpr: Значение TPR (True Positive Rate)
        fdr: Значение FDR (False Discovery Rate)
        font_scale: Масштаб шрифта
        thickness: Толщина линий
        position: Позиция текста (x, y) в пикселях
        
    Returns:
        Изображение с нарисованными метриками
    """
    result_image = image.copy()
    
    # Формирование текста метрик
    tpr_text = f"TPR: {tpr:.3f}"
    fdr_text = f"FDR: {fdr:.3f}"
    
    # Вычисление размеров текста
    (tpr_width, tpr_height), tpr_baseline = cv2.getTextSize(
        tpr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    (fdr_width, fdr_height), fdr_baseline = cv2.getTextSize(
        fdr_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Вычисление размеров фона
    max_width = max(tpr_width, fdr_width)
    total_height = tpr_height + fdr_height + 10  # 10 пикселей между строками
    
    x, y = position
    
    # Рисуем полупрозрачный фон для читаемости
    overlay = result_image.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - tpr_height - 5),
        (x + max_width + 5, y + total_height + 5),
        (0, 0, 0),  # Черный цвет
        -1
    )
    cv2.addWeighted(overlay, 0.6, result_image, 0.4, 0, result_image)
    
    # Отрисовка TPR
    cv2.putText(
        result_image,
        tpr_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),  # Зеленый цвет для TPR
        thickness,
        cv2.LINE_AA
    )
    
    # Отрисовка FDR
    cv2.putText(
        result_image,
        fdr_text,
        (x, y + tpr_height + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),  # Красный цвет для FDR
        thickness,
        cv2.LINE_AA
    )
    
    return result_image

