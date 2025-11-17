import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class ModelConfig:
    name: str
    model_path: str
    classes: List[str]
    input_size: Tuple[int, int]
    description: str = ""
    vehicle_classes: List[str] = None  # Это поле теперь будет заполнено
    config_path: Optional[str] = None
    scale_factor: float = 1.0
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    swap_rb: bool = True
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5


class BaseDetector(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.net = cv2.dnn.readNet(config.model_path, config.config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # **ИСПРАВЛЕНИЕ:** Выносим логику определения целевых классов в базовый класс.
        # Мы заранее вычисляем set() индексов (ID) тех классов,
        # которые нам интересны (из 'vehicle_classes').

        # 1. Создаем set() имен целевых классов (для быстрого поиска)
        target_class_names = set(self.config.vehicle_classes or [])

        # 2. Создаем set() ID целевых классов
        self.target_class_ids = set()
        for i, class_name in enumerate(self.config.classes):
            if class_name in target_class_names:
                self.target_class_ids.add(i)

        print(f"[Detector {config.name}] Target classes: {target_class_names}")
        print(f"[Detector {config.name}] Target class IDs: {self.target_class_ids}")

    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            self.config.scale_factor,
            self.config.input_size,
            self.config.mean,
            swapRB=self.config.swap_rb,
            crop=False
        )
        return blob

    @abstractmethod
    def postprocess(self, outputs, image_shape) -> List[Detection]:
        pass

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        boxes = []
        confidences = []
        class_ids = []  # NMS можно применять по-классно, но для простоты - общий

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # NMSBoxes ожидает (x, y, w, h)
            confidences.append(det.confidence)
            class_ids.append(det.class_id)

        # Применяем NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.config.confidence_threshold,
            self.config.nms_threshold
        )

        if len(indices) > 0:
            result = []
            # .flatten() для обработки случаев, когда indices = [[0], [2], ...]
            for i in indices.flatten():
                result.append(detections[i])
            return result

        return []

    def detect(self, image) -> List[Detection]:
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward()  # Стандартный forward()
        detections = self.postprocess(outputs, image.shape[:2])
        return self._apply_nms(detections)