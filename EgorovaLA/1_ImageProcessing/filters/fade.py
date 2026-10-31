import numpy as np

from .base import ImageFilter
from .utils import to_gray, to_uint8


class FadeColor(ImageFilter):
    """Выцветание."""

    name = "fade"

    def __init__(self, saturation=0.5, fade=0.4):
        self.saturation = saturation
        self.fade = fade

    def apply_filter(self, image):
        img = image.astype(np.float32)
        gray = to_gray(img)[..., None]
        img = gray + self.saturation * (img - gray)

        # сжатие диапазона [0, 255] -> [low, high]
        low = 90 * self.fade
        high = 255 - 35 * self.fade
        return to_uint8(low + img * (high - low) / 255)
