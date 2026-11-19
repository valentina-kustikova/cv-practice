import numpy as np

from .base import ImageFilter
from .utils import blur, convolve, to_gray, to_uint8


class Neon(ImageFilter):
    name = "neon"

    # цвета в порядке B, G, R
    COLORS = {
        "pink": (200, 60, 255),
        "blue": (255, 150, 0),
        "green": (80, 255, 80),
        "yellow": (0, 230, 255),
    }

    # ядра оператора Собеля
    SOBEL_X = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

    SOBEL_Y = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]

    def __init__(self, color="pink", glow=1.5, border=1.0):
        if color not in self.COLORS:
            raise ValueError(f"color может быть: {', '.join(self.COLORS)}")
        self.color = np.array(self.COLORS[color], dtype=np.float32)
        self.glow = float(glow)
        self.border = float(border)

    def apply_filter(self, image):
        h, w = image.shape[:2]
        step = max(1, max(h, w) // 400)

        line_radius = max(1,int(step * self.border))

        gray = to_gray(blur(image, 2 * step))

        gx = convolve(gray, self.SOBEL_X)
        gy = convolve(gray, self.SOBEL_Y)
        edges = np.sqrt(gx ** 2 + gy ** 2)

        edges = edges / np.percentile(edges, 99)
        edges = np.clip((edges - 0.3) / 0.7, 0, 1)[:, :, np.newaxis]
 
        lines = blur(edges * self.color, line_radius)

        glow_radius = max(1, int(step * self.border * self.glow))
        glow = blur(lines, 4 * step) + blur(lines, 12 * step) + blur(lines, 30 * step)

        background = image.astype(np.float32) * 0.3
        result = background + lines + self.glow * glow
        return to_uint8(result)