import numpy as np

from .base import ImageFilter
from .utils import blur, screen, to_uint8


class InfraredFilm(ImageFilter):
    """Инфракрасная плёнка."""

    name = "infrared"

    def __init__(self, mode="color", glow=0.4):
        if mode not in ("color", "bw"):
            raise ValueError("mode должен быть color или bw")
        self.mode = mode
        self.glow = glow

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        h, w = img.shape[:2]

        # оценка ИК: зелень яркая, небо тёмное
        foliage = np.maximum(2 * g - r - b, 0)
        sky = np.maximum(b - (r + g) / 2, 0)
        ir = np.clip(0.9 * r + 0.2 * g - 0.2 * b + 1.2 * foliage - 0.7 * sky, 0, 255)

        if self.mode == "color":
            film = np.dstack([g, r, ir])
        else:
            film = np.dstack([ir, ir, ir])

        # ореол вокруг светлых участков
        light = np.clip(film - 160, 0, None) * 255 / 95
        film = screen(film, self.glow * blur(light, max(h, w) / 80))
        return to_uint8(film)
