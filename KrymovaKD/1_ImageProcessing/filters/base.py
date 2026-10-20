
from abc import ABC, abstractmethod


class ImageFilter(ABC):
    """Базовый класс для всех фильтров изображений."""

    @abstractmethod
    def apply(self, image, **kwargs):
        """Применяет фильтр к изображению."""
        pass

    @staticmethod
    def get_filter(name):
        """Возвращает объект фильтра по его имени."""
        from .geometry import Resize
        from .color import Grayscale, Antique, FadeColor, Film
        from .effects import Matte, Noise, Neon

        filters = {
            "resize": Resize,
            "grayscale": Grayscale,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "noise": Noise,
            "neon": Neon,
        }

        if name not in filters:
            available = ", ".join(filters.keys())
            raise ValueError(
                f"Неизвестный фильтр: {name}. Доступны: {available}"
            )

        return filters[name]()
