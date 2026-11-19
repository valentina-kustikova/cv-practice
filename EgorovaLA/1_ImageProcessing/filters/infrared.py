import numpy as np

from .base import ImageFilter
from .utils import to_uint8


class InfraredFilm(ImageFilter):
    name = "infrared"

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        leaves = np.maximum(2 * g - r - b, 0)
        sky = np.maximum(b - (r + g) / 2, 0)
        ir = r + 1.2 * leaves - 0.7 * sky

        result = np.dstack([g, r, ir])
        return to_uint8(result)
