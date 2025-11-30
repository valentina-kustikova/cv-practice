from dataclasses import dataclass
from typing import List, Tuple, Sequence
import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Detection:
    """Результат детектирования"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class DetectorConfig:
    name: str
    model_path: str
    config_path: str
    classes: List[str]
    input_size: Tuple[int, int]
    scale: float
    swap_rb: bool
    mean: Tuple[float, float, float]
    conf_threshold: float
    nms_threshold: float


class BaseDetector(ABC):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.net = cv.dnn.readNet(config.model_path, config.config_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def _make_blob(self, image):
        blob = cv.dnn.blobFromImage(
            image,
            scalefactor=self.config.scale,
            size=self.config.input_size,
            mean=self.config.mean,
            swapRB=self.config.swap_rb,
            crop=False
        )
        return blob

    def _forward(self, blob: np.ndarray):
        self.net.setInput(blob)
        return self.net.forward()

    @abstractmethod
    def _postprocess(self, outputs: np.ndarray, image_shape: Tuple[int, int]) -> List[Detection]:
        """Постобработка"""
        pass

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Non-Maximum Suppression"""
        if not detections:
            return []

        boxes = []
        confidences = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(det.confidence))

        indices = cv.dnn.NMSBoxes(
            boxes, confidences,
            self.config.conf_threshold,
            self.config.nms_threshold
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [detections[i] for i in indices]

    def detect(self, image) -> List[Detection]:
        """Цикл детектирования"""
        blob = self._make_blob(image)
        outputs = self._forward(blob)
        raw_detections = self._postprocess(outputs, image.shape[:2])
        return self._apply_nms(raw_detections)
