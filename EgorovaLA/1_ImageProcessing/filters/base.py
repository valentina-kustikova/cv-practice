from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):
    # Базовый класс фильтра

    name = ""

    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        # Принимает изображение BGR uint8, возвращает новое изображение

    @staticmethod
    def names() -> list[str]:
        return [cls.name for cls in ImageFilter.__subclasses__()]

    @staticmethod
    def get_filter(name: str, **params) -> "ImageFilter":
        # Создаёт фильтр по имени
        for cls in ImageFilter.__subclasses__():
            if cls.name == name:
                try:
                    return cls(**params)
                except TypeError as exc:
                    raise ValueError(f"неверные параметры фильтра '{name}': {exc}") from None
        raise ValueError(f"неизвестный фильтр '{name}', доступны: {', '.join(ImageFilter.names())}")
