
import numpy as np

from .base import ImageFilter


class Resize(ImageFilter):
    """Изменяет разрешение изображения методом ближайшего соседа."""

    def apply(self, image, width=None, height=None, scale_x=0.5, scale_y=0.5):
        old_height, old_width = image.shape[:2]

        if width is None:
            width = int(old_width * scale_x)
        if height is None:
            height = int(old_height * scale_y)

        if width <= 0 or height <= 0:
            raise ValueError("Ширина и высота должны быть положительными")

        y_indices = (
            np.arange(height) * old_height / height
        ).astype(int)
        x_indices = (
            np.arange(width) * old_width / width
        ).astype(int)

        y_indices = np.clip(y_indices, 0, old_height - 1)
        x_indices = np.clip(x_indices, 0, old_width - 1)

        return image[y_indices[:, None], x_indices[None, :]]
