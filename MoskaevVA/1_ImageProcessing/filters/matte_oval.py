import numpy as np

from .base import ImageFilter


class Matte(ImageFilter):
    name = "matte"

    def __init__(self, softness: float = 0.15, scale: float = 0.9,
                 mask_width: float = None, mask_height: float = None,
                 center_x: float = None, center_y: float = None):

        self.softness = float(softness)
        self.scale = float(scale)
        self.mask_width = None if mask_width is None else float(mask_width)
        self.mask_height = None if mask_height is None else float(mask_height)
        self.center_x = None if center_x is None else float(center_x)
        self.center_y = None if center_y is None else float(center_y)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        cx = self.center_x if self.center_x is not None else (w - 1) / 2.0
        cy = self.center_y if self.center_y is not None else (h - 1) / 2.0

        a = (self.mask_width if self.mask_width is not None
             else w / 2.0 * self.scale)
        b = (self.mask_height if self.mask_height is not None
             else h / 2.0 * self.scale)
        a = max(a, 1e-3)
        b = max(b, 1e-3)

        d = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)

        s = max(self.softness, 1e-3)
        t = np.clip((1.0 - d) / s + 0.5, 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)   # smoothstep

        out = (image.astype(np.float32) * mask[..., None]
               + 255.0 * (1 - mask[..., None]))
        return self._u8(out)