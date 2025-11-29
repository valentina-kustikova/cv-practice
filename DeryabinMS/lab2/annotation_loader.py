import os
from typing import Dict, List, Tuple


class AnnotationLoader:
    """
    Загружаем разметку
    Формат строки:
    image_id class_name x1 y1 x2 y2 ...
    Используем только объекты класса 'car'
    """

    def __init__(self, annotation_path: str, target_class: str = "car"):
        self.annotation_path = annotation_path
        self.target_class = target_class.lower()
        self._annotations: Dict[str, List[Tuple[int, int, int, int]]] = self._load()

    def _load(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        annotations: Dict[str, List[Tuple[int, int, int, int]]] = {}

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                image_id_raw = parts[0]
                class_name = parts[1].lower()

                if class_name != self.target_class:
                    continue

                x1, y1, x2, y2 = map(int, parts[2:6])

                image_id = image_id_raw.zfill(6)

                if image_id not in annotations:
                    annotations[image_id] = []

                annotations[image_id].append((x1, y1, x2, y2))

        return annotations

    def get_ground_truth(self, image_id: str) -> List[Tuple[int, int, int, int]]:
        """Возвращаем список GT-боксов кадра"""
        return self._annotations.get(image_id, [])
