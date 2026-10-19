"""Набор фильтров изображений на NumPy и OpenCV.

OpenCV используется для чтения и записи изображений. Основные преобразования
выполняются вручную над массивами NumPy, в том числе масштабирование: cv2.resize
не используется.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Type

import numpy as np


ImageArray = np.ndarray


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Ограничивает значения диапазоном интенсивностей 0..255 и переводит в uint8."""
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _gray_bgr(image: ImageArray) -> ImageArray:
    """Переводит BGR-изображение в серое по формуле яркости, без cv2.cvtColor."""
    image_float = image.astype(np.float32)
    blue = image_float[..., 0]
    green = image_float[..., 1]
    red = image_float[..., 2]
    # Взвешенная яркость учитывает чувствительность зрения к разным каналам.
    gray = 0.114 * blue + 0.587 * green + 0.299 * red
    return _to_uint8(gray)


def _box_blur_3x3(array: np.ndarray) -> np.ndarray:
    """Простое усреднение соседних 3x3 значений (ручная свёртка)."""
    height, width = array.shape
    padded = np.pad(array.astype(np.float32), ((1, 1), (1, 1)), mode="edge")
    result = np.zeros((height, width), dtype=np.float32)

    # Складываем девять сдвинутых копий массива — это свёртка с ядром 1/9.
    for dy in range(3):
        for dx in range(3):
            result += padded[dy:dy + height, dx:dx + width]
    return result / 9.0


class ImageFilter(ABC):
    """Базовый класс фильтра: каждый наследник реализует apply_filter()."""

    @abstractmethod
    def apply_filter(self, image: ImageArray) -> ImageArray:
        """Применяет фильтр и возвращает новое изображение."""

    @staticmethod
    def get_filter(name: str, **params: object) -> "ImageFilter":
        """Фабричный метод: создаёт нужный фильтр по строковому имени."""
        try:
            filter_class = FILTER_REGISTRY[name.lower()]
        except KeyError as exc:
            available = ", ".join(sorted(FILTER_REGISTRY))
            raise ValueError(f"Неизвестный фильтр {name!r}. Доступны: {available}") from exc
        return filter_class(**params)

    @staticmethod
    def validate_strength(strength: float) -> float:
        """Проверяет параметр силы эффекта, который должен лежать в диапазоне [0, 1]."""
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Параметр strength должен быть в диапазоне от 0 до 1.")
        return strength


class Resize(ImageFilter):
    """Изменение размера изображения без вызова cv2.resize()."""

    def __init__(self, width: int, height: int, interpolation: str = "bilinear") -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Ширина и высота должны быть положительными числами.")
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation должен быть 'nearest' или 'bilinear'.")
        self.width = int(width)
        self.height = int(height)
        self.interpolation = interpolation

    def apply_filter(self, image: ImageArray) -> ImageArray:
        source_height, source_width = image.shape[:2]
        if self.interpolation == "nearest":
            # Для каждого пикселя результата выбирается ближайший пиксель источника.
            x_indices = np.minimum(
                (np.arange(self.width) * source_width / self.width).astype(int),
                source_width - 1,
            )
            y_indices = np.minimum(
                (np.arange(self.height) * source_height / self.height).astype(int),
                source_height - 1,
            )
            return image[y_indices[:, None], x_indices[None, :]].copy()

        # Билинейная интерполяция: пиксель результата — взвешенная сумма
        # четырёх ближайших пикселей оригинального изображения.
        x = (np.arange(self.width) + 0.5) * source_width / self.width - 0.5
        y = (np.arange(self.height) + 0.5) * source_height / self.height - 0.5
        x = np.clip(x, 0, source_width - 1)
        y = np.clip(y, 0, source_height - 1)

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, source_width - 1)
        y1 = np.minimum(y0 + 1, source_height - 1)
        wx = x - x0
        wy = y - y0

        top_left = image[y0[:, None], x0[None, :]].astype(np.float32)
        top_right = image[y0[:, None], x1[None, :]].astype(np.float32)
        bottom_left = image[y1[:, None], x0[None, :]].astype(np.float32)
        bottom_right = image[y1[:, None], x1[None, :]].astype(np.float32)

        if image.ndim == 3:
            wx = wx[None, :, None]
            wy = wy[:, None, None]
        else:
            wx = wx[None, :]
            wy = wy[:, None]

        top = top_left * (1.0 - wx) + top_right * wx
        bottom = bottom_left * (1.0 - wx) + bottom_right * wx
        return _to_uint8(top * (1.0 - wy) + bottom * wy)


class RGB2GrayScale(ImageFilter):
    """Преобразует цветное BGR-изображение в оттенки серого."""

    def apply_filter(self, image: ImageArray) -> ImageArray:
        return _gray_bgr(image)


class Antique(ImageFilter):
    """Сепия с мягким затемнением краёв — стилизация под старую фотографию."""

    def __init__(self, strength: float = 0.8) -> None:
        self.strength = self.validate_strength(strength)

    def apply_filter(self, image: ImageArray) -> ImageArray:
        src = image.astype(np.float32)
        blue, green, red = src[..., 0], src[..., 1], src[..., 2]

        # Матрица сепии. Порядок каналов на входе и выходе OpenCV — B, G, R.
        sepia = np.stack(
            (
                0.272 * blue + 0.534 * green + 0.131 * red,
                0.349 * red + 0.686 * green + 0.168 * blue,
                0.393 * red + 0.769 * green + 0.189 * blue,
            ),
            axis=-1,
        )

        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        dx = (xx - (width - 1) / 2) / max(width / 2, 1)
        dy = (yy - (height - 1) / 2) / max(height / 2, 1)
        # Виньетка: края немного темнее центра.
        vignette = 1.0 - 0.18 * np.clip(dx * dx + dy * dy, 0.0, 1.0)
        sepia *= vignette[..., None]

        result = src * (1.0 - self.strength) + sepia * self.strength
        return _to_uint8(result)


class FadeColor(ImageFilter):
    """Выцветание: ослабление контраста, поднятие теней и уменьшение насыщенности."""

    def __init__(self, strength: float = 0.6) -> None:
        self.strength = self.validate_strength(strength)

    def apply_filter(self, image: ImageArray) -> ImageArray:
        src = image.astype(np.float32)
        gray = _gray_bgr(image).astype(np.float32)[..., None]
        # Слегка уменьшаем насыщенность, смешивая оригинал с серым изображением.
        desaturated = 0.82 * src + 0.18 * gray
        # Светлая подложка поднимает тёмные тона и делает изображение бледнее.
        faded = 0.72 * desaturated + 0.28 * 235.0
        result = src * (1.0 - self.strength) + faded * self.strength
        return _to_uint8(result)


class InfraredFilm(ImageFilter):
    """Стилизация под инфракрасную плёнку с псевдоцветным преобразованием каналов."""

    def __init__(self, strength: float = 0.8) -> None:
        self.strength = self.validate_strength(strength)

    def apply_filter(self, image: ImageArray) -> ImageArray:
        src = image.astype(np.float32)
        blue, green, red = src[..., 0], src[..., 1], src[..., 2]

        # Оценка яркости сцены с акцентом на зелёный канал; это художественная
        # имитация, а не восстановление настоящего инфракрасного спектра.
        infrared_luminance = 0.72 * green + 0.28 * red
        pseudo_infrared = np.stack(
            (
                0.12 * blue,
                0.72 * infrared_luminance + 8.0,
                1.10 * infrared_luminance + 18.0,
            ),
            axis=-1,
        )
        result = src * (1.0 - self.strength) + pseudo_infrared * self.strength
        return _to_uint8(result)


class Matte(ImageFilter):
    """Белая овальная маска с плавным переходом к краям изображения."""

    def __init__(self, strength: float = 1.0, softness: float = 0.28) -> None:
        self.strength = self.validate_strength(strength)
        if not 0.01 <= softness < 1.0:
            raise ValueError("softness должен быть в диапазоне от 0.01 до значения меньше 1.")
        self.softness = float(softness)

    def apply_filter(self, image: ImageArray) -> ImageArray:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius_x = max(width / 2.0, 1.0)
        radius_y = max(height / 2.0, 1.0)

        # r=0 соответствует центру овала; r около 1 — его границе.
        radius = np.sqrt(((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2)
        inner_radius = 1.0 - self.softness
        alpha = np.clip((radius - inner_radius) / self.softness, 0.0, 1.0)
        alpha = (alpha * self.strength)[..., None]

        src = image.astype(np.float32)
        result = src * (1.0 - alpha) + 255.0 * alpha
        return _to_uint8(result)


class ScratchesAndNoise(ImageFilter):
    """Добавляет зернистость и случайные светлые/тёмные царапины."""

    def __init__(
        self,
        noise_strength: float = 12.0,
        scratch_count: int = 40,
        seed: int = 42,
    ) -> None:
        if noise_strength < 0:
            raise ValueError("noise_strength не может быть отрицательным.")
        if scratch_count < 0:
            raise ValueError("scratch_count не может быть отрицательным.")
        self.noise_strength = float(noise_strength)
        self.scratch_count = int(scratch_count)
        self.seed = int(seed)

    def apply_filter(self, image: ImageArray) -> ImageArray:
        rng = np.random.default_rng(self.seed)
        # Аддитивный гауссов шум: N(0, sigma^2).
        noisy = image.astype(np.float32) + rng.normal(
            loc=0.0, scale=self.noise_strength, size=image.shape
        )
        result = _to_uint8(noisy)
        height, width = result.shape[:2]
        min_dimension = max(1, min(height, width))
        min_length = max(3, int(min_dimension * 0.05))
        max_length = max(min_length + 1, int(min_dimension * 0.45))

        # Рисуем каждую царапину как набор пиксельных координат отрезка.
        # Большинство царапин почти вертикальны, как на старой плёнке.
        for _ in range(self.scratch_count):
            x0 = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height))
            length = int(rng.integers(min_length, max_length))
            angle = float(rng.uniform(-0.14, 0.14))
            dx = int(round(math.sin(angle) * length))
            dy = int(round(math.cos(angle) * length))
            steps = max(abs(dx), abs(dy), 1) + 1
            xs = np.rint(np.linspace(x0, x0 + dx, steps)).astype(int)
            ys = np.rint(np.linspace(y0, y0 + dy, steps)).astype(int)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs, ys = xs[valid], ys[valid]

            # Случайно выбирается светлая или тёмная царапина.
            shade = int(rng.choice([25, 235]))
            thickness = int(rng.choice([1, 1, 2]))
            for offset in range(thickness):
                y_thick = np.clip(ys + offset, 0, height - 1)
                result[y_thick, xs] = (shade, shade, shade)

        return result


class Neon(ImageFilter):
    """Неоновый эффект: градиенты Собеля, затемнение фона и цветное свечение контуров."""

    COLORS = {
        # Значения заданы в порядке BGR, принятом OpenCV.
        "cyan": (255, 95, 35),
        "magenta": (220, 35, 255),
        "green": (70, 255, 55),
    }

    def __init__(self, strength: float = 0.9, color: str = "cyan") -> None:
        self.strength = self.validate_strength(strength)
        if color not in self.COLORS:
            raise ValueError(f"Неизвестный цвет неона: {color}. Доступны: {', '.join(self.COLORS)}")
        self.color = color

    def apply_filter(self, image: ImageArray) -> ImageArray:
        gray = _gray_bgr(image).astype(np.float32)
        height, width = gray.shape
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")

        # Ручное применение операторов Собеля по горизонтали и вертикали.
        gx = (
            -padded[:-2, :-2] + padded[:-2, 2:]
            - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
            - padded[2:, :-2] + padded[2:, 2:]
        )
        gy = (
            -padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
            + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        )
        magnitude = np.sqrt(gx * gx + gy * gy)

        # Порог зависит от конкретного изображения: остаются наиболее выраженные границы.
        threshold = max(35.0, float(np.percentile(magnitude, 85)))
        edges = (magnitude >= threshold).astype(np.float32)

        # Несколько простых усреднений распространяют свет от контуров в соседние пиксели.
        glow = edges.copy()
        for _ in range(5):
            glow = _box_blur_3x3(glow)
        max_glow = float(glow.max())
        if max_glow > 0:
            glow /= max_glow

        neon_color = np.asarray(self.COLORS[self.color], dtype=np.float32)
        base = image.astype(np.float32) * (1.0 - 0.85 * self.strength)
        halo = glow[..., None] * neon_color[None, None, :] * (0.9 * self.strength)
        bright_edges = edges[..., None] * neon_color[None, None, :] * (1.5 * self.strength)
        return _to_uint8(base + halo + bright_edges)


FILTER_REGISTRY: dict[str, Type[ImageFilter]] = {
    "resize": Resize,
    "grayscale": RGB2GrayScale,
    "antique": Antique,
    "fade": FadeColor,
    "film": InfraredFilm,
    "matte": Matte,
    "scratches": ScratchesAndNoise,
    "neon": Neon,
}
