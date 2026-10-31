import numpy as np

from .base import ImageFilter
from .utils import to_uint8


class Antique(ImageFilter):
    #Сепия

    name = "antique"

    # матрица сепии для порядка каналов B, G, R
    SEPIA = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ], dtype=np.float32)

    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def apply_filter(self, image):
        img = image.astype(np.float32)
        sepia = np.clip(img @ self.SEPIA.T, 0, 255)
        k = self.intensity
        return to_uint8((1 - k) * img + k * sepia)
