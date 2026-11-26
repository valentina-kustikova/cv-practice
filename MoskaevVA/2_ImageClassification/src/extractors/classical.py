from __future__ import annotations
from abc import abstractmethod
import cv2
import numpy as np
from .base import FeatureExtractor

class ClassicalExtractor(FeatureExtractor):
    """Базовый класс классических локальных экстракторов."""
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features

    @abstractmethod
    def detect_and_compute(self, image):
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def _gray(self, image):
        if image is None:
            raise ValueError("Передано пустое изображение")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

class SIFTExtractor(ClassicalExtractor):
    """SIFT: float-дескрипторы, совместимые с евклидовым K-means."""
    def __init__(self, max_features: int = 1000):
        super().__init__(max_features)
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT недоступен в установленной сборке OpenCV.")
        self.detector = cv2.SIFT_create(nfeatures=max_features)

    def detect_and_compute(self, image):
        _, desc = self.detector.detectAndCompute(self._gray(image), None)
        return np.asarray(desc, dtype=np.float32) if desc is not None else np.empty((0, 128), np.float32)

    def get_config(self):
        return {"name": "sift", "max_features": self.max_features}

class KAZEExtractor(ClassicalExtractor):
    """Классический детектор ORB (замена KAZE для OpenCV 5.0)."""
    def __init__(self, max_features: int = 1000):
        super().__init__(max_features)
        self.detector = cv2.ORB_create(nfeatures=max_features)

    def detect_and_compute(self, image):
        _, desc = self.detector.detectAndCompute(self._gray(image), None)
        if desc is None:
            return np.empty((0, 256), np.float32)
        return np.unpackbits(desc, axis=1).astype(np.float32)

    def get_config(self):
        return {"name": "kaze", "max_features": self.max_features}
