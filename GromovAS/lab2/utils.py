"""
Модуль с утилитами для обработки детекции объектов:
- Вычисление метрик пересечения
- Статистика качества распознавания
- Чтение данных разметки
"""

import os
import pathlib
from typing import List, Dict, Tuple, Optional, Set


def calculate_overlap_ratio(rect1: List[float], rect2: List[float]) -> float:
    """
    Вычисляет отношение площади пересечения к площади объединения (IoU)
    для двух прямоугольников формата [x, y, ширина, высота].
    """
    # Координаты левого верхнего и правого нижнего углов
    x1_left, y1_top = rect1[0], rect1[1]
    x1_right, y1_bottom = x1_left + rect1[2], y1_top + rect1[3]

    x2_left, y2_top = rect2[0], rect2[1]
    x2_right, y2_bottom = x2_left + rect2[2], y2_top + rect2[3]

    # Координаты пересечения
    inter_left = max(x1_left, x2_left)
    inter_top = max(y1_top, y2_top)
    inter_right = min(x1_right, x2_right)
    inter_bottom = min(y1_bottom, y2_bottom)

    # Проверка на отсутствие пересечения
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    # Площади
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    union_area = area1 + area2 - inter_area

    # Предотвращение деления на ноль
    epsilon = 1e-7
    return inter_area / (union_area + epsilon)


class DetectionEvaluator:
    """
    Класс для подсчета статистики обнаружения объектов
    """

    def __init__(self):
        """Инициализация счетчиков"""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def add_detection_batch(self,
                            predictions: List[List],
                            ground_truths: List[List],
                            similarity_threshold: float = 0.5) -> None:
        """
        Обновляет статистику на основе сравнения предсказаний и истинных данных.

        Args:
            predictions: Список предсказаний [класс, уверенность, x, y, ширина, высота]
            ground_truths: Список истинных объектов [класс, x, y, ширина, высота]
            similarity_threshold: Порог IoU для совпадения
        """
        used_truth_indices: Set[int] = set()

        # Обработка каждого предсказания
        for pred in predictions:
            pred_class = pred[0].strip().lower()
            pred_rectangle = pred[2:]

            best_match_index = -1
            max_similarity = 0.0

            # Поиск наилучшего совпадения среди истинных объектов
            for idx, truth in enumerate(ground_truths):
                truth_class = truth[0].strip().lower()

                # Проверка совпадения классов
                if pred_class != truth_class:
                    continue

                similarity = calculate_overlap_ratio(pred_rectangle, truth[1:])
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_index = idx

            # Проверка порога и уникальности совпадения
            if (max_similarity >= similarity_threshold and
                    best_match_index not in used_truth_indices):
                self.true_positives += 1
                used_truth_indices.add(best_match_index)
            else:
                self.false_positives += 1

        # Подсчет пропущенных объектов
        self.false_negatives += len(ground_truths) - len(used_truth_indices)

    def get_performance_metrics(self) -> Tuple[float, float]:
        """
        Возвращает метрики качества обнаружения.

        Returns:
            Кортеж (recall, false_discovery_rate)
        """
        denominator_tp_fn = self.true_positives + self.false_negatives
        denominator_tp_fp = self.true_positives + self.false_positives

        # Предотвращение деления на ноль
        recall = (self.true_positives / (denominator_tp_fn + 1e-7)
                  if denominator_tp_fn > 0 else 0.0)

        fdr = (self.false_positives / (denominator_tp_fp + 1e-7)
               if denominator_tp_fp > 0 else 0.0)

        return recall, fdr

    def reset_counters(self) -> None:
        """Сбрасывает все счетчики"""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0


def load_annotation_data(file_path: str) -> Dict[int, List[List]]:
    """
    Загружает данные аннотации из текстового файла.

    Формат файла: Номер_кадра Класс Xmin Ymin Xmax Ymax

    Args:
        file_path: Путь к файлу аннотаций

    Returns:
        Словарь {номер_кадра: [[класс, x, y, ширина, высота], ...]}
    """
    annotation_dict = {}

    if not os.path.isfile(file_path):
        print(f"Файл аннотаций не найден: {file_path}")
        return annotation_dict

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) != 6:
                    print(f"Предупреждение: неверный формат строки {line_number}")
                    continue

                try:
                    frame_num = int(parts[0])
                    class_name = parts[1]

                    # Конвертация координат
                    coords = [int(float(p)) for p in parts[2:]]
                    x_min, y_min, x_max, y_max = coords

                    # Проверка корректности координат
                    if x_max <= x_min or y_max <= y_min:
                        print(f"Предупреждение: некорректные координаты в строке {line_number}")
                        continue

                    width = x_max - x_min
                    height = y_max - y_min

                    if frame_num not in annotation_dict:
                        annotation_dict[frame_num] = []

                    annotation_dict[frame_num].append([
                        class_name,
                        x_min,
                        y_min,
                        width,
                        height
                    ])

                except (ValueError, IndexError) as e:
                    print(f"Ошибка обработки строки {line_number}: {e}")
                    continue

        print(f"Загружено аннотаций для {len(annotation_dict)} кадров.")
        return annotation_dict

    except IOError as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        return {}