import numpy as np

from .base import ImageFilter


class Neon(ImageFilter):
    name = "neon"

    def __init__(self, threshold: float = 30.0, color=(0, 255, 255),
                 glow_radius: int = 2, glow_strength: float = 1.2):
        self.threshold = float(threshold)
        self.color = np.array(color, dtype=np.float32)
        self.glow_radius = int(max(0, glow_radius))
        self.glow_strength = float(glow_strength)

    def _laplacian(self, gray: np.ndarray) -> np.ndarray:
        k = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float32)
        h, w = gray.shape
        p = np.pad(gray, 1, mode="edge")
        out = np.zeros_like(gray, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if k[dy, dx] != 0:
                    out += k[dy, dx] * p[dy:dy + h, dx:dx + w]
        return out

    def _blur(self, img: np.ndarray, radius: int) -> np.ndarray:
        out = img.copy()
        for _ in range(radius):
            p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
            out = (p[:-2, 1:-1] + p[2:, 1:-1] +
                   p[1:-1, :-2] + p[1:-1, 2:] +
                   p[1:-1, 1:-1]) / 5.0
        return out

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        lap = self._laplacian(gray)
        edges = np.clip((np.abs(lap) - self.threshold) * 3.0, 0, 255)

        neon = (edges[..., None] / 255.0) * self.color

        out = img * 0.2 + neon
        if self.glow_radius > 0:
            glow = self._blur(neon, self.glow_radius)
            out += glow * self.glow_strength

        return self._u8(out)