from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class BoundingBox:
    frame_id: int
    class_name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_xyxy(self):
        return self.x_min, self.y_min, self.x_max, self.y_max


class DataReader:
    def __init__(
        self,
        images_dir,
        annotation_path,
        image_template = "{:06d}.jpg",
    ):
        self.images_dir = Path(images_dir)
        self.annotation_path = Path(annotation_path)
        self.image_template = image_template

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Директория с изображениями не найдена: {self.images_dir}")

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Файл разметки не найден: {self.annotation_path}")

        self.annotations = {}
        self._frame_ids = []

        self._load_annotations()
        self._frame_ids = sorted(self.annotations.keys())

    def _load_annotations(self):
        annotations = {}

        with self.annotation_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 6:
                    raise ValueError(
                        f"Неверный формат строки в разметке (строка {line_no}): {line}"
                    )

                frame_id_str, class_name, x1_str, y1_str, x2_str, y2_str = parts
                try:
                    frame_id = int(frame_id_str)
                    x_min = int(x1_str)
                    y_min = int(y1_str)
                    x_max = int(x2_str)
                    y_max = int(y2_str)
                except ValueError as e:
                    raise ValueError(
                        f"Не удалось преобразовать координаты в целое число (строка {line_no}): {line}"
                    ) from e

                box = BoundingBox(
                    frame_id=frame_id,
                    class_name=class_name,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                )

                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append(box)

        self.annotations = annotations

    def __len__(self):
        return len(self._frame_ids)

    def frame_ids(self):
        return list(self._frame_ids)

    def get_frame_path(self, frame_id):
        return self.images_dir / self.image_template.format(frame_id)

    def get_boxes(self, frame_id):
        return self.annotations.get(frame_id, [])

    def load_image(self, frame_id):
        img_path = self.get_frame_path(frame_id)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
        return img

    def __getitem__(self, idx):
        frame_id = self._frame_ids[idx]
        image = self.load_image(frame_id)
        boxes = self.get_boxes(frame_id)
        return frame_id, image, boxes

    def __iter__(self):
        for frame_id in self._frame_ids:
            yield frame_id, self.load_image(frame_id), self.get_boxes(frame_id)
