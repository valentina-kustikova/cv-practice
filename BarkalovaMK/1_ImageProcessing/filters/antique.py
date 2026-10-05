import numpy as np
from .base import ImageFilter


class Antique(ImageFilter):
    """Эффект 'антиквариат': сепия + виньетка + зерно."""

    def __init__(self, vignette: float = 0.6, noise: float = 6.0):
        self.vignette = vignette
        self.noise = noise

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)

        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        sr = 0.393 * r + 0.769 * g + 0.189 * b
        sg = 0.349 * r + 0.686 * g + 0.168 * b
        sb = 0.272 * r + 0.534 * g + 0.131 * b
        sepia = np.stack([sb, sg, sr], axis=2)

        h, w = sepia.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        mask = np.clip(1 - self.vignette * d, 0, 1)[..., None]
        sepia *= mask

        if self.noise > 0:
            sepia += np.random.normal(0, self.noise, sepia.shape)

        return np.clip(sepia, 0, 255).astype(np.uint8)