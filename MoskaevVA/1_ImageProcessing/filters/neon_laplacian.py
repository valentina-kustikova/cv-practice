import numpy as np

from .base import ImageFilter
from .grayscale import RGB2GrayScale


class Neon(ImageFilter):
    name = "neon"

    def __init__(self, neon_strength: float = 1.5, neon_blur: float = 0.8):
        if neon_strength < 0:
            raise ValueError("neon-strength должен быть >= 0")
        if neon_blur < 0:
            raise ValueError("neon-blur должен быть >= 0")
        self.neon_strength = float(neon_strength)
        self.neon_blur = float(neon_blur)

    @staticmethod
    def _blur(image: np.ndarray, iterations: int) -> np.ndarray:
        # Размытие усреднением 3x3
        result = image.astype(np.float32)

        for _ in range(iterations):
            p = np.pad(result, 1, mode="edge")
            result = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0

        return result

    def apply_filter(self, img: np.ndarray) -> np.ndarray:
        gray = RGB2GrayScale().apply_filter(img).astype(np.float32)[..., 0]

        # Поиск границ разностями соседних пикселей
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        gy[:-1, :] = gray[1:, :] - gray[:-1, :]

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        edges = np.clip(magnitude / 255.0, 0.0, 1.0)

        # Два слоя свечения.
        # neon_blur - ширина ореола.
        soft_iterations = max(0, round(4 * self.neon_blur))
        wide_iterations = max(0, round(10 * self.neon_blur))

        soft_glow = self._blur(edges, soft_iterations)
        wide_glow = self._blur(edges, wide_iterations)

        # neon_strength - яркость свечения.
        glow = self.neon_strength * (
            edges * 70.0
            + soft_glow * 130.0
            + wide_glow * 180.0
        )

        result = img.astype(np.float32).copy()

        # RGB: синий + зеленый = красивый
        result[:, :, 1] += glow
        result[:, :, 2] += glow

        return np.clip(result, 0, 255).astype(np.uint8)