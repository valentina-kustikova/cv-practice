import numpy as np

from .base import ImageFilter
from .utils import blur, screen, to_gray, to_uint8


class Neon(ImageFilter):
    """Неоновые контуры."""

    name = "neon"

    COLORS = {"cyan": (255, 255, 0), "pink": (180, 60, 255), "green": (60, 255, 60), "yellow": (0, 240, 255)}

    def __init__(self, color="rainbow", threshold=0.12, glow=1.2, background=0.15):
        if color not in ("rainbow", "source", *self.COLORS):
            raise ValueError(f"color должен быть одним из: rainbow, source, {', '.join(self.COLORS)}")
        self.color = color
        self.threshold = threshold
        self.glow = glow
        self.background = background

    def apply_filter(self, image):
        img = image.astype(np.float32)

        # оператор Собеля
        p = np.pad(blur(to_gray(img), 1), 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = np.clip(mag / (np.percentile(mag, 99) or 1), 0, 1)
        edges = np.clip((mag - self.threshold) / (1 - self.threshold), 0, 1)[..., None]

        # цвет линий
        if self.color == "rainbow":
            a = np.arctan2(gy, gx)
            tint = 127.5 * (1 + np.dstack([np.cos(a + 2.094), np.cos(a - 2.094), np.cos(a)]))
        elif self.color == "source":
            low, high = img.min(axis=2, keepdims=True), img.max(axis=2, keepdims=True)
            tint = (img - low) / np.maximum(high - low, 1) * 255
        else:
            tint = np.array(self.COLORS[self.color], np.float32)

        # свечение
        lines = tint * edges
        halo = 0.9 * blur(lines, 2) + 0.7 * blur(lines, 6) + 0.5 * blur(lines, 14)
        neon = np.clip(lines + self.glow * halo, 0, 255)
        return to_uint8(screen(img * self.background, neon))
