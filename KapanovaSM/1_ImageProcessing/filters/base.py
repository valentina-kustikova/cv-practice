# Абстрактный базовый класс ImageFilter

from abc import ABC, abstractmethod


class ImageFilter(ABC):
    # Базовый класс для всех фильтров

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> "ImageFilter":
        # создаёт нужный фильтр по имени
        from .resize import Resize
        from .grayscale import ToGray
        from .antique import Antique
        from .fade import FadeColor
        from .film import Film
        from .matte import Matte
        from .aged import Aged
        from .neon import Neon

        registry = {
            "resize": Resize,
            "gray": ToGray,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "aged": Aged,
            "neon": Neon,
        }
        key = filter_type.lower()
        if key not in registry:
            raise ValueError(
                f"Неизвестный фильтр '{filter_type}'. "
                f"Доступные: {', '.join(registry.keys())}"
            )
        return registry[key](**kwargs)

    @abstractmethod
    def apply_filter(self, image):
        # Применить фильтр к изображению
        raise NotImplementedError