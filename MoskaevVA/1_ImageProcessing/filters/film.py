import numpy as np

from .base import ImageFilter


class InfraredFilm(ImageFilter):
    name = "film"

    def __init__(self, mix: float = 0.6, grain: float = 0.3, seed: int = 42):
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.grain = float(max(0.0, grain))
        self.seed = int(seed)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        m = self.mix

        ir = g * 1.3
        out = img.copy()
        out[..., 0] = r * (1 - m) + ir * m
        out[..., 1] = g * (1 - m) + b * m * 0.2
        out[..., 2] = b * (1 - m) + (255.0 - ir) * m * 0.3

        if self.grain > 0:
            rng = np.random.default_rng(self.seed)
            lum = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]) / 255.0
            mask = np.clip(1.0 - 4.0 * (lum - 0.5) ** 2, 0.1, 1.0)
            noise = rng.normal(0.0, self.grain * 25.0, out.shape[:2]).astype(np.float32)
            out += noise[..., None] * mask[..., None]

        return self._u8(out)