import numpy as np
from .base import ImageFilter


class RGB2GrayScale(ImageFilter):
#Перевод в оттенки серого

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()
       #разделение каналов
        b = image[:, :, 0].astype(np.float32) #все пиксели нулевого канала
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)