import numpy as np

from .antique import Antique
from .base import ImageFilter
from .utils import to_uint8


class OldPhoto(ImageFilter):
    name = "oldphoto"

    def __init__(self, noise=20, scratches=20):
        self.noise = float(noise)
        self.scratches = int(scratches)

    def apply_filter(self, image):
        img = Antique().apply_filter(image).astype(np.float32)
        h, w = img.shape[:2]

        grain = np.random.normal(0, self.noise, (h, w))
        img = img + grain[:, :, np.newaxis]

        for _ in range(self.scratches):
            x0 = np.random.randint(0, w)
            y0 = np.random.randint(0, h // 2)
            length = np.random.randint(h // 10, h // 5)
            slope = np.random.uniform(-1, 1)
            bright = np.random.randint (60, 120)
            for t in range(length):
                y = y0 + t
                x = int (x0 + slope * t)
                if 0 <= x < w and 0 <= y < h:
                    img[y,x]+=bright

        return to_uint8(img)