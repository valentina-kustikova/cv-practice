from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):

    @staticmethod
    def get_filter(name, param=None):
        if name == "gray":
            return GrayScale()
        if name == "resize":
            return Resize(param)
        raise ValueError(f"Неизвестный фильтр: {name}")

    @abstractmethod
    def apply_filter(self, image):
        pass


class GrayScale(ImageFilter):

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return np.clip(gray, 0, 255).astype(np.uint8)

class Resize(ImageFilter):
    def __init__(self, scale=None):
        if scale is None:
            scale = 0.5
        if scale <= 0:
            raise ValueError("Масштаб должен быть больше 0")
        self.scale = scale
    def apply_filter(self, image):
        height, width = image.shape[:2]

        new_height = int(height * self.scale)
        new_width = int(width * self.scale)

        rows = (np.arange(new_height) / self.scale).astype(int)
        cols = (np.arange(new_width) / self.scale).astype(int)

        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)

        return image[rows][:, cols]


# TODO (этап 4): Resize, RGB2GrayScale
# TODO (этап 5): Antique, FadeColor, FilmEffect
# TODO (этап 6): Matte
# TODO (этап 7): OldPhoto (царапины и шум)
# TODO (этап 8): Neon
