import numpy as np
from .base import ImageFilter


class Resize(ImageFilter):
#Изменение разрешения (билинейная интерполяция)

    def __init__(self, scale: float = 0.5):
        if scale <= 0:
            raise ValueError("scale должен быть > 0")
        self.scale = scale

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        #новый размер
        new_h = max(1, int(h * self.scale))
        new_w = max(1, int(w * self.scale))
        #коорд новых пикселей в исх изобр
        y_new = np.linspace(0, h - 1, new_h)
        x_new = np.linspace(0, w - 1, new_w)
        x_grid, y_grid = np.meshgrid(x_new, y_new)
        #4 соседа
        x0 = np.floor(x_grid).astype(np.int32)
        y0 = np.floor(y_grid).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        #веса интерполяции
        wx = x_grid - x0
        wy = y_grid - y0

        if image.ndim == 2:
            top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
            bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
            result = (1 - wy) * top + wy * bottom
        else:
            wx = wx[..., None]
            wy = wy[..., None]
            top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
            bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
            result = (1 - wy) * top + wy * bottom

        return np.clip(result, 0, 255).astype(np.uint8)