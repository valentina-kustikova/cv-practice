from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class FeatureExtractor(ABC):
    """Общий интерфейс извлечения признаков."""

    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        """Возвращает матрицу дескрипторов формы (число признаков, размерность)."""
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def create(name: str, **kwargs) -> "FeatureExtractor":
        name = name.lower()
        if name in {"sift", "kaze"}:
            from .classical import SIFTExtractor, KAZEExtractor
            return {"sift": SIFTExtractor, "kaze": KAZEExtractor}[name](**kwargs)
        if name in {"resnet18", "convnext_tiny"}:
            from .cnn import ResNet18Extractor, ConvNeXtTinyExtractor
            return {
                "resnet18": ResNet18Extractor,
                "convnext_tiny": ConvNeXtTinyExtractor,
            }[name](**kwargs)

        raise ValueError(f"Неизвестный экстрактор: {name}")
