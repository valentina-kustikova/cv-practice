from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


Image = np.ndarray


def _clip_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError("Цвет должен иметь формат R,G,B, например 255,220,180.")
    rgb = tuple(int(x) for x in parts)
    if any(x < 0 or x > 255 for x in rgb):
        raise ValueError("Компоненты цвета должны находиться в диапазоне 0..255.")
    return rgb


def manual_resize(
    image: Image,
    width: int,
    height: int,
    interpolation: str = "bilinear",
) -> Image:
    if width <= 0 or height <= 0:
        raise ValueError("width и height должны быть положительными.")

    src_h, src_w = image.shape[:2]

    if width == src_w and height == src_h:
        return image.copy()

    y = np.linspace(0.0, src_h - 1.0, height)
    x = np.linspace(0.0, src_w - 1.0, width)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    if interpolation == "nearest":
        yi = np.rint(yy).astype(np.int32)
        xi = np.rint(xx).astype(np.int32)
        return image[yi, xi].copy()

    if interpolation != "bilinear":
        raise ValueError("interpolation: nearest или bilinear.")

    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = (yy - y0)[..., None]
    wx = (xx - x0)[..., None]

    top_left = image[y0, x0].astype(np.float32)
    top_right = image[y0, x1].astype(np.float32)
    bottom_left = image[y1, x0].astype(np.float32)
    bottom_right = image[y1, x1].astype(np.float32)

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    result = top * (1.0 - wy) + bottom * wy
    return _clip_u8(result)


class ImageFilter(ABC):
    name = "base"

    @staticmethod
    def get_filter(name: str, **params) -> "ImageFilter":
        registry = {
            "resize": Resize,
            "grayscale": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "noise": ScratchesAndNoise,
            "neon": Neon,
        }
        if name not in registry:
            raise ValueError(f"Неизвестный фильтр: {name}")
        return registry[name](**params)

    @abstractmethod
    def apply_filter(self, image: Image) -> Image:
        raise NotImplementedError


@dataclass
class Resize(ImageFilter):
    width: int
    height: int
    interpolation: str = "bilinear"
    name = "resize"

    def apply_filter(self, image: Image) -> Image:
        return manual_resize(image, self.width, self.height, self.interpolation)


class RGB2GrayScale(ImageFilter):
    name = "grayscale"

    def apply_filter(self, image: Image) -> Image:
        b, g, r = cv2.split(image.astype(np.float32))
        gray = 0.1140 * b + 0.5870 * g + 0.2990 * r
        return _clip_u8(gray)


@dataclass
class Antique(ImageFilter):
    strength: float = 1.0
    name = "antique"

    def apply_filter(self, image: Image) -> Image:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("antique strength должен быть от 0 до 1.")

        b, g, r = cv2.split(image.astype(np.float32))
        # Каналы рассчитываются матрицами коэффициентов сепии.
        out_r = 0.393 * r + 0.769 * g + 0.189 * b
        out_g = 0.349 * r + 0.686 * g + 0.168 * b
        out_b = 0.272 * r + 0.534 * g + 0.131 * b

        sepia = np.stack((out_b, out_g, out_r), axis=-1)
        return _clip_u8(
            image.astype(np.float32) * (1.0 - self.strength)
            + sepia * self.strength
        )


@dataclass
class FadeColor(ImageFilter):
    alpha: float = 0.35
    color: Tuple[int, int, int] = (235, 215, 175)  # RGB
    name = "fade"

    def apply_filter(self, image: Image) -> Image:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("fade alpha должен быть от 0 до 1.")

        r, g, b = self.color
        # OpenCV хранит изображение в BGR, поэтому цвет разворачиваем.
        target = np.array([b, g, r], dtype=np.float32)
        result = image.astype(np.float32) * (1.0 - self.alpha) + target * self.alpha
        return _clip_u8(result)


@dataclass
class Film(ImageFilter):
    gamma: float = 0.85
    grain: float = 0.06
    seed: int = 42
    name = "film"

    def apply_filter(self, image: Image) -> Image:
        if self.gamma <= 0:
            raise ValueError("gamma должен быть > 0.")
        if not 0.0 <= self.grain <= 1.0:
            raise ValueError("grain должен быть от 0 до 1.")

        x = image.astype(np.float32) / 255.0

        b, g, r = cv2.split(x)
        film_b = 0.95 * b + 0.05 * g
        film_g = 0.12 * b + 0.78 * g + 0.10 * r
        film_r = 0.08 * g + 0.92 * r
        film = np.stack((film_b, film_g, film_r), axis=-1)

        film = np.power(np.clip(film, 0.0, 1.0), self.gamma)

        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.grain / 3.0, size=film.shape[:2])[..., None]
        film = film + noise

        # Легкое усиление контраста относительно среднего.
        film = 0.5 + (film - 0.5) * 1.10
        return _clip_u8(film * 255.0)


@dataclass
class Matte(ImageFilter):
    radius: float = 0.58
    strength: float = 1.0
    softness: float = 2.2
    name = "matte"

    def apply_filter(self, image: Image) -> Image:
        if not 0.20 < self.radius < 1.0:
            raise ValueError("matte radius должен быть в диапазоне 0.2..1.0.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("matte strength должен быть от 0 до 1.")
        if self.softness <= 0:
            raise ValueError("matte softness должен быть > 0.")

        h, w = image.shape[:2]
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, h),
            np.linspace(-1.0, 1.0, w),
            indexing="ij",
        )

        radius_map = np.sqrt(xx * xx + yy * yy)
        t = np.clip((radius_map - self.radius) / (1.0 - self.radius), 0.0, 1.0)
        edge = np.power(t, self.softness)[..., None] * self.strength

        white = np.full_like(image, 255, dtype=np.float32)
        result = image.astype(np.float32) * (1.0 - edge) + white * edge
        return _clip_u8(result)


@dataclass
class ScratchesAndNoise(ImageFilter):
    noise: float = 0.10
    scratch_density: float = 0.03
    seed: int = 42
    name = "noise"

    def apply_filter(self, image: Image) -> Image:
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("noise должен быть от 0 до 1.")
        if not 0.0 <= self.scratch_density <= 0.5:
            raise ValueError("scratch_density должен быть от 0 до 0.5.")

        h, w = image.shape[:2]
        rng = np.random.default_rng(self.seed)

        grain = rng.normal(
            0.0, 255.0 * self.noise * 0.18, size=(h, w, 1)
        )

        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )

        phase1 = rng.uniform(0.0, 2.0 * np.pi)
        phase2 = rng.uniform(0.0, 2.0 * np.pi)
        stripe_field = (
            0.5 * (np.sin(2.0 * np.pi * xx / max(w / 13.0, 2.0) + phase1) > 0.997)
            + 0.5 * (np.sin(2.0 * np.pi * xx / max(w / 29.0, 2.0) + phase2) > 0.998)
        )

        specks = rng.random((h, w)) < (self.scratch_density * 0.04)
        scratch = np.clip(stripe_field + specks, 0.0, 1.0)[..., None]

        # Царапины делаем преимущественно светлыми, иногда затемняем.
        scratch_value = rng.uniform(0.65, 1.0, size=(h, w, 1))
        result = (
            image.astype(np.float32)
            + grain
            + scratch * (255.0 * scratch_value - image.astype(np.float32))
        )
        return _clip_u8(result)


@dataclass
class Neon(ImageFilter):
    strength: float = 1.5
    blur: float = 0.8
    name = "neon"

    def apply_filter(self, image: Image) -> Image:
        if self.strength < 0:
            raise ValueError("neon strength должен быть >= 0.")
        if self.blur < 0:
            raise ValueError("neon blur должен быть >= 0.")

        b, g, r = cv2.split(image.astype(np.float32))
        gray = 0.1140 * b + 0.5870 * g + 0.2990 * r
        gray_u8 = _clip_u8(gray)

        if self.blur > 0:
            k = max(3, int(round(self.blur * 6)) | 1)
            gray_u8 = cv2.GaussianBlur(gray_u8, (k, k), self.blur)

        kernel_x = np.array(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=np.float32,
        )
        kernel_y = kernel_x.T

        gx = cv2.filter2D(gray_u8.astype(np.float32), cv2.CV_32F, kernel_x)
        gy = cv2.filter2D(gray_u8.astype(np.float32), cv2.CV_32F, kernel_y)

        magnitude = np.sqrt(gx * gx + gy * gy)
        magnitude /= max(float(magnitude.max()), 1.0)
        edge = np.clip(magnitude * self.strength, 0.0, 1.0)[..., None]

        dark = image.astype(np.float32) * (1.0 - 0.78 * edge)
        neon_bgr = np.concatenate(
            [
                edge * 255.0,
                edge * 190.0,
                edge * 255.0,
            ],
            axis=-1,
        )

        result = dark * 0.65 + neon_bgr * 0.95
        return _clip_u8(result)


def read_image(filename: str) -> Image:
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"Не удалось прочитать изображение: {filename}. "
            "Проверьте путь, расширение и доступность файла."
        )
    return image


def save_image(filename: str, image: Image) -> None:
    ok = cv2.imwrite(filename, image)
    if not ok:
        raise IOError(f"Не удалось сохранить изображение: {filename}")
