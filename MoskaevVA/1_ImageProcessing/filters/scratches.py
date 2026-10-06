import numpy as np

from .base import ImageFilter


class Scratches(ImageFilter):
    name = "scratches"

    def __init__(self, n_scratches: int = 8, noise_sigma: float = 12.0,
                 seed: int = 42, dust: float = 0.004, vertical_bias: float = 0.8):
        self.n_scratches = int(n_scratches)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        self.dust = float(max(0.0, dust))
        self.vertical_bias = float(np.clip(vertical_bias, 0.0, 1.0))

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]
        out = image.astype(np.float32)

        # 1) зерно по всему изображению
        out += rng.normal(0, self.noise_sigma, out.shape).astype(np.float32)

        # 2) "мусор" — белые и чёрные точки
        if self.dust > 0:
            rand = rng.random((h, w))
            white = rand < self.dust * 0.6
            black = (rand >= self.dust * 0.6) & (rand < self.dust)
            out[white] = 245.0
            out[black] = 15.0

        # 3) царапины с затуханием
        for _ in range(self.n_scratches):
            if rng.random() < self.vertical_bias:
                # вертикальная царапина сверху вниз
                x0 = int(rng.integers(0, w))
                y0 = int(rng.integers(0, max(1, h // 3)))
                length = int(rng.integers(h // 2, h))
                angle = float(rng.uniform(-0.1, 0.1))
            else:
                # произвольная короткая
                x0 = int(rng.integers(0, w))
                y0 = int(rng.integers(0, h))
                length = int(rng.integers(h // 6, h // 3))
                angle = float(rng.uniform(-0.5, 0.5))

            thick = int(rng.integers(1, 3))
            alpha = np.linspace(1.0, 0.25, max(length, 1), dtype=np.float32)

            for t in range(length):
                y = y0 + t
                x = int(x0 + angle * t)
                if 0 <= y < h and 0 <= x < w:
                    a = alpha[t]
                    xs = slice(x, min(w, x + thick))
                    out[y, xs] = 255.0 * a + out[y, xs] * (1.0 - a)

        return self._u8(out)