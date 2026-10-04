"""Библиотека фильтров изображений."""

from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):
    """Абстрактный базовый класс фильтра."""

    @staticmethod
    def get_filter(name, **params):
        """Фабричный метод: возвращает объект нужного фильтра по имени."""
        # TODO (этап 3)
        pass

    @abstractmethod
    def apply_filter(self, image):
        """Применяет фильтр к изображению и возвращает результат."""
        pass


# TODO (этап 4): Resize, RGB2GrayScale
# TODO (этап 5): Antique, FadeColor, FilmEffect
# TODO (этап 6): Matte
# TODO (этап 7): OldPhoto (царапины и шум)
# TODO (этап 8): Neon
