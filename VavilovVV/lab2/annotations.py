import os
from typing import List, Tuple

class AnnotationLoader:
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> dict[str, List[Tuple[int, int, int, int]]]:
        annotations: dict[str, List[Tuple[int, int, int, int]]] = {}
        if not os.path.exists(self.annotation_path):
            return annotations

        with open(self.annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[1].lower() == "car":
                    image_id = parts[0].zfill(6)
                    x1, y1, x2, y2 = map(int, parts[2:6])
                    if image_id not in annotations:
                        annotations[image_id] = []
                    annotations[image_id].append((x1, y1, x2, y2))

        return annotations

    def get_ground_truth(self, image_id: str) -> List[Tuple[int, int, int, int]]:
        return self.annotations.get(image_id, [])