import numpy as np

from .base import ImageFilter
from .utils import to_gray, to_uint8


class FadeColor(ImageFilter):
    name = "fade"

    def __init__(self, saturation=0.5, fade=0.3):
        self.saturation = float(saturation)
        self.fade = float(fade)

    def apply_filter(self, image):
        img = image.astype(np.float32)
        gray = to_gray(img)[:, :, np.newaxis]

        img = gray + self.saturation * (img - gray)
        img = (1 - self.fade) * img + self.fade * 128
        return to_uint8(img)
