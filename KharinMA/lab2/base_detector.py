from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_xyxy(self):
        return self.x_min, self.y_min, self.x_max, self.y_max


class BaseDetector(ABC):
    _registry = {}

    def __init_subclass__(cls, *, model_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if model_name is not None:
            BaseDetector._registry[model_name] = cls

    def __init__(
        self,
        conf_threshold=0.5,
        nms_threshold=0.4,
        class_names=None,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or []
        self.net = None

        self._class_colors = {}

    @classmethod
    def available_models(cls):
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name, **kwargs):
        try:
            detector_cls = cls._registry[name]
        except KeyError as e:
            raise ValueError(
                f"Неизвестная модель детектора '{name}'. "
                f"Доступные: {list(cls._registry.keys())}"
            ) from e
        return detector_cls(**kwargs)

    def detect(self, image):
        if self.net is None:
            raise RuntimeError("Сеть не загружена: self.net is None")

        blob, meta = self._preprocess(image)
        outputs = self._forward(blob)
        detections = self._postprocess(image, outputs)
        return detections

    def draw_detections(
        self,
        image,
        detections,
        show_label=True,
    ):
        img = image.copy()

        for det in detections:
            color = self._class_colors.get(det.class_id)
            if color is None:
                color = tuple(int(c)
                              for c in np.random.randint(0, 255, size=3))
                self._class_colors[det.class_id] = color

            cv2.rectangle(
                img,
                (det.x_min, det.y_min),
                (det.x_max, det.y_max),
                color,
                thickness=2,
            )

            if show_label:
                label = f"{det.class_name} {det.confidence:.3f}"
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                y1 = max(det.y_min - th - baseline, 0)
                cv2.rectangle(
                    img,
                    (det.x_min, y1),
                    (det.x_min + tw, y1 + th + baseline),
                    color,
                    thickness=-1,
                )
                cv2.putText(
                    img,
                    label,
                    (det.x_min, y1 + th),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        return img

    @abstractmethod
    def _preprocess(self, image):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, blob):
        raise NotImplementedError

    @abstractmethod
    def _postprocess(
        self,
        image,
        outputs,
        meta=None,
    ):
        raise NotImplementedError