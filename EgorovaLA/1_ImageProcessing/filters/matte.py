import numpy as np

from .base import ImageFilter
from .utils import to_uint8


class Matte(ImageFilter):
    name = "matte"

    COLORS = {"white": 255, "black": 0}

    def __init__(self, size=1.0, softness=0.3, color="white"):
        if color not in self.COLORS:
            raise ValueError("color может быть только white или black")
        self.size = float(size)
        self.softness = float(softness)
        self.color = self.COLORS[color]

    def apply_filter(self, image):
        h, w = image.shape[:2]
        center_x = w / 2
        center_y = h / 2
        y, x = np.mgrid[0:h, 0:w]

        # d = 1 на границе овала, меньше 1 внутри
        dist = np.sqrt(((x - center_x) / center_x) ** 2 + ((y - center_y) / center_y) ** 2) / self.size

        mask = (dist - (1 - self.softness)) / self.softness
        mask = np.clip(mask, 0, 1)[:, :, np.newaxis]

        result = (1 - mask) * image + mask * self.color
        return to_uint8(result)
