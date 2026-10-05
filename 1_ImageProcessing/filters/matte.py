import numpy as np
from .base import ImageFilter


class Matte(ImageFilter):
    """Эффект маски: овальная рамка, края белые."""

    def __init__(self, border: int = 40, softness: float = 0.15):
        self.border = border
        self.softness = softness

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)

        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        ax = max(cx - self.border, 1)
        ay = max(cy - self.border, 1)

        d = np.sqrt(((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2)
        mask = np.clip((1 - d) / max(self.softness, 1e-6), 0, 1)[..., None]

        out = mask * img + (1 - mask) * 255
        return np.clip(out, 0, 255).astype(np.uint8)