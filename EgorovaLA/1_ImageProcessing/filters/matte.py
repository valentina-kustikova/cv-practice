import numpy as np

from .base import ImageFilter
from .utils import to_uint8


class Matte(ImageFilter):
    """Овальная рамка с плавным краем."""

    name = "matte"

    def __init__(self, size=1.05, softness=0.3, color=255):
        self.size = size
        self.softness = max(softness, 1e-6)
        self.color = color

    def apply_filter(self, image):
        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # расстояние до центра: 1 на границе овала
        d = np.sqrt(((x - w / 2) / (self.size * w / 2)) ** 2 + ((y - h / 2) / (self.size * h / 2)) ** 2)
        t = np.clip((d - 1 + self.softness) / self.softness, 0, 1)
        alpha = (t * t * (3 - 2 * t))[..., None]
        return to_uint8((1 - alpha) * image + alpha * self.color)
