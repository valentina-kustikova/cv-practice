from abc import ABC, abstractmethod
import numpy as np


class ImageFilter(ABC):

    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет фильтр к изображению."""
        pass

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> 'ImageFilter':
        """Фабричный метод для создания экземпляров фильтров."""
        from .processing_filters import (
            ResizeFilter,
            GrayScaleFilter,
            AntiqueFilter,
            FadeColorFilter,
            InfraredFilmFilter,
            MatteFilter,
            OldPhotoNoiseFilter,
            NeonFilter,
        )

        mapping = {
            'resize': ResizeFilter,
            'gray': GrayScaleFilter,
            'antique': AntiqueFilter,
            'fade': FadeColorFilter,
            'infrared': InfraredFilmFilter,
            'matte': MatteFilter,
            'old_photo': OldPhotoNoiseFilter,
            'neon': NeonFilter,
        }

        filter_cls = mapping.get(filter_type.lower())
        if filter_cls is None:
            raise ValueError(f"Неизвестный тип фильтра: {filter_type}")
        return filter_cls(**kwargs)