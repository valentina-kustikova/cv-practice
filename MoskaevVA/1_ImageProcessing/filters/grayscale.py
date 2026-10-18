import numpy as np

from .base import ImageFilter


class RGB2GrayScale(ImageFilter):
    name = "grayscale"

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        y = np.clip(y, 0, 255).astype(np.uint8)
        return np.stack([y, y, y], axis=-1)