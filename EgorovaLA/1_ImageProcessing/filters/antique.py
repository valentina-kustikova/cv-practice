import numpy as np

from .base import ImageFilter
from .matte import Matte
from .utils import to_uint8


class Antique(ImageFilter):
    name = "antique"

    def __init__(self, intensity=1.0):
        self.intensity = float(intensity)

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_b = 0.272 * r + 0.534 * g + 0.131 * b

        # собираем обратно в порядке B, G, R
        sepia = np.dstack([new_b, new_g, new_r])
        sepia = np.clip(sepia, 0, 255)

        k = self.intensity
        result = to_uint8((1 - k) * img + k * sepia)

        vignette = Matte(size=1.5, softness=0.5, color="black")

        return vignette.apply_filter(result)
