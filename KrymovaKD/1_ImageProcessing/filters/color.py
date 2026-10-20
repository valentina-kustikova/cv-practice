import numpy as np

from .base import ImageFilter


class Grayscale(ImageFilter):
    """Преобразует изображение в оттенки серого."""

    def apply(self, image, **kwargs):
        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)

        gray = 0.114 * blue + 0.587 * green + 0.299 * red

        return np.clip(gray, 0, 255).astype(np.uint8)


class Antique(ImageFilter):
    """Придаёт изображению тёплый оттенок сепии."""

    def apply(self, image, **kwargs):
        blue = image[:, :, 0].astype(np.float32)
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 2].astype(np.float32)

        result = np.empty_like(image, dtype=np.float32)

        result[:, :, 2] = 0.393 * red + 0.769 * green + 0.189 * blue
        result[:, :, 1] = 0.349 * red + 0.686 * green + 0.168 * blue
        result[:, :, 0] = 0.272 * red + 0.534 * green + 0.131 * blue

        return np.clip(result, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    """Имитирует выцветание изображения заданным цветом."""

    def apply(self, image, alpha=0.35, color=(235, 215, 175), **kwargs):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha должен быть от 0 до 1")

        fade_color = np.array(color[::-1], dtype=np.float32)
        original = image.astype(np.float32)

        result = (1 - alpha) * original + alpha * fade_color

        return np.clip(result, 0, 255).astype(np.uint8)



class Film(ImageFilter):
    """Имитирует тёплую цветовую передачу плёнки и добавляет зерно."""

    def apply(self, image, strength=0.7, noise=8, **kwargs):
        if not 0 <= strength <= 1:
            raise ValueError("strength должен быть от 0 до 1")
        if noise < 0:
            raise ValueError("noise не может быть отрицательным")

        result = image.astype(np.float32).copy()

        blue = result[:, :, 0].copy()
        green = result[:, :, 1].copy()
        red = result[:, :, 2].copy()

        # Слегка уменьшаем синий канал и усиливаем красный.
        result[:, :, 0] = blue * (1 - 0.25 * strength)
        result[:, :, 1] = green * (1 - 0.05 * strength)
        result[:, :, 2] = red * (1 + 0.15 * strength)

        # Добавляем плёночное зерно.
        rng = np.random.default_rng()
        grain = rng.normal(0, noise, image.shape).astype(np.float32)
        result += grain

        return np.clip(result, 0, 255).astype(np.uint8)
