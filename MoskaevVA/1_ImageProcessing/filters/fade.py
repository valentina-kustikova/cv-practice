import numpy as np

from .base import ImageFilter


class FadeColor(ImageFilter):
    name = "fade"

    def __init__(self, strength: float = 0.5, black_lift: int = 30):
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.black_lift = int(black_lift)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        # 1) десатурация
        gray = img.mean(axis=2, keepdims=True)
        out = img * (1 - self.strength) + gray * self.strength

        # 2) сжатие динамического диапазона [f_min, f_max]
        f_min = float(self.black_lift)
        f_max = 255.0 - f_min
        if f_max > f_min:
            out = f_min + out * (f_max - f_min) / 255.0
        else:
            out = np.full_like(out, f_min)

        return self._u8(out)