import numpy as np
from .base import ImageFilter


class RGB2GrayScale(ImageFilter):
    """Перевод в оттенки серого (ITU-R BT.601)."""

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()

        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)