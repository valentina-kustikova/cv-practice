import numpy as np

from .base import ImageFilter
from .utils import to_uint8


class Resize(ImageFilter):
    """Изменение размера билинейной интерполяцией."""

    name = "resize"

    def __init__(self, width=None, height=None, scale=0.5):
        self.width = width
        self.height = height
        self.scale = scale

    def apply_filter(self, image):
        h, w = image.shape[:2]
        k = self.scale
        if self.width:
            k = self.width / w
        if self.height:
            k = self.height / h
        new_w = self.width or max(1, round(w * k))
        new_h = self.height or max(1, round(h * k))

        # координаты центров новых пикселей в исходном изображении
        ys = np.clip((np.arange(new_h) + 0.5) * h / new_h - 0.5, 0, h - 1)
        xs = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5, 0, w - 1)
        y0, x0 = ys.astype(int), xs.astype(int)
        y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
        dy = (ys - y0)[:, None, None]
        dx = (xs - x0)[None, :, None]

        img = image.astype(np.float32)
        top = img[y0][:, x0] * (1 - dx) + img[y0][:, x1] * dx
        bottom = img[y1][:, x0] * (1 - dx) + img[y1][:, x1] * dx
        return to_uint8(top * (1 - dy) + bottom * dy)
