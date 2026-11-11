import numpy as np

from .base import ImageFilter


class Antique(ImageFilter):
    name = "antique"

    def __init__(self, strength: float = 1.0, grain: float = 0.25, seed: int = 42):
        self.strength = float(strength)
        self.grain = float(max(0.0, grain))
        self.seed = int(seed)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        sepia = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ], dtype=np.float32)
        out = img.reshape(-1, 3) @ sepia.T
        out = out.reshape(img.shape)

        # виньетка
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d /= d.max() + 1e-6
        v = 1.0 - 0.6 * d * self.strength
        out *= v[..., None]

        # зерно, сильнее в тенях
        if self.grain > 0:
            rng = np.random.default_rng(self.seed)
            lum = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]) / 255.0
            mask = np.clip(1.0 - 4.0 * (lum - 0.5) ** 2, 0.1, 1.0)
            noise = rng.normal(0.0, self.grain * 20.0, out.shape[:2]).astype(np.float32)
            out += noise[..., None] * mask[..., None]

        return self._u8(out)