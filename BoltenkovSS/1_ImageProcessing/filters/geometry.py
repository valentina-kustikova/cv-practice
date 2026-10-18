import numpy as np

from filters.base import ImageFilter


class Resize(ImageFilter):
    def apply_filter(self, img, **kwargs):
        scale_x = kwargs.get("scale_x", 0.5)
        scale_y = kwargs.get("scale_y", 0.5)
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("scale_x and scale_y must be positive")

        h, w = img.shape[:2]
        new_h, new_w = int(h * scale_y), int(w * scale_x)
        if new_h < 1 or new_w < 1:
            raise ValueError(
                f"Scale factors are too small: result would be {new_w}x{new_h} pixels"
            )

        y_indices = (np.arange(new_h) / scale_y).astype(int)
        x_indices = (np.arange(new_w) / scale_x).astype(int)

        y_indices = np.clip(y_indices, 0, h - 1)
        x_indices = np.clip(x_indices, 0, w - 1)

        return img[y_indices[:, None], x_indices]
