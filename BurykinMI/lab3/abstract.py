from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np


# ============================================================================
# Содержит абстрактные базовые классы для архитектуры приложения:
# - FeatureExtractor: интерфейс для извлечения признаков из изображений
# - ClassificationStrategy: интерфейс для стратегий классификации (паттерн Strategy)
# Определяет контракты для методов extract, train, predict, save/load
# ============================================================================

class FeatureExtractor(ABC):
    """Абстрактный класс для извлечения признаков из изображений"""

    @abstractmethod
    def extract(self, image_path: str) -> np.ndarray:
        """Извлечь признаки из изображения"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Получить название экстрактора"""
        pass


class ClassificationStrategy(ABC):
    """Абстрактная стратегия классификации"""

    @abstractmethod
    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Обучить модель"""
        pass

    @abstractmethod
    def predict(self, image_path: str) -> str:
        """Предсказать класс изображения"""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Сохранить модель"""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Загрузить модель"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Получить параметры модели"""
        pass
