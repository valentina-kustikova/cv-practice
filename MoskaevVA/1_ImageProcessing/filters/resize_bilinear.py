import numpy as np

from .base import ImageFilter


class Resize(ImageFilter):
    name = "resize"

    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("width/height must be positive")
        self.width = int(width)
        self.height = int(height)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        sh, sw = image.shape[:2]
        dw, dh = self.width, self.height

        x = np.clip((np.arange(dw) + 0.5) * (sw / dw) - 0.5, 0, sw - 1)
        y = np.clip((np.arange(dh) + 0.5) * (sh / dh) - 0.5, 0, sh - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, sw - 1)
        y1 = np.clip(y0 + 1, 0, sh - 1)

        wx = (x - x0).astype(np.float32)[None, :, None]
        wy = (y - y0).astype(np.float32)[:, None, None]

        img = image.astype(np.float32)
        top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
        bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
        out = top * (1 - wy) + bot * wy
        return self._u8(out)