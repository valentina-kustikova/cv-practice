import numpy as np

from .base import ImageFilter


class Matte(ImageFilter):
    name = "matte"

    def __init__(self, softness: float = 0.15, scale: float = 0.9):
        self.softness = float(softness)
        self.scale = float(scale)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        a = w / 2.0 * self.scale
        b = h / 2.0 * self.scale
        d = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)

        s = max(self.softness, 1e-3)
        t = np.clip((1.0 - d) / s + 0.5, 0.0, 1.0)
        # smoothstep: 3t^2 - 2t^3 — мягче, чем линейный переход
        mask = t * t * (3.0 - 2.0 * t)

        out = image.astype(np.float32) * mask[..., None] + 255.0 * (1 - mask[..., None])
        return self._u8(out)