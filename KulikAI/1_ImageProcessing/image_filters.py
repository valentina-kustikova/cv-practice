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
    strength: float = 0.35
    noise: float = 8.0
    name = "film"
    def apply_filter(self, image: Image) -> Image:
        img = image.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)

        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        new_r = (1 - self.strength) * r + self.strength * b
        new_g = g * (1 - 0.15 * self.strength)
        new_b = (1 - self.strength) * b + self.strength * r
        out = np.stack([new_b, new_g, new_r], axis=2)
        out[..., 0] = np.clip(out[..., 0] * 1.05, 0, 255)
        out[..., 2] = np.clip(out[..., 2] * 0.95, 0, 255) 

        if self.noise > 0:
            out += np.random.normal(0, self.noise, out.shape)

        return np.clip(out, 0, 255).astype(np.uint8)


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

        grain = rng.normal(0.0, 255.0 * self.noise * 0.18, size=(h, w, 1))

        stripe_field = np.zeros((h, w), dtype=np.float32)
        
        num_scratches = int(w * self.scratch_density * 0.5) 
        
        for _ in range(num_scratches):
            x1 = rng.integers(0, w)
            y1 = rng.integers(0, h)
            
            length = rng.integers(int(h * 0.05), int(h * 0.40))
            
            angle = rng.uniform(-5, 5) * np.pi / 180.0
            
            x2 = int(x1 + length * np.sin(angle))
            y2 = int(y1 + length * np.cos(angle))
            
            thickness = int(rng.choice([1, 2], p=[0.8, 0.2]))
            intensity = rng.uniform(0.3, 0.7)
            
            cv2.line(stripe_field, (x1, y1), (x2, y2), intensity, thickness)

        # 3. Пятна/Точки (мелкая грязь)
        specks = rng.random((h, w)) < (self.scratch_density * 0.04)
        
        # Объединяем царапины и точки
        scratch = np.clip(stripe_field + specks.astype(np.float32), 0.0, 1.0)[..., None]

        scratch_value = rng.uniform(0.65, 1.0, size=(h, w, 1))
        result = (
            image.astype(np.float32)
            + grain
            + scratch * (255.0 * scratch_value - image.astype(np.float32))
        )
        result = _clip_u8(result)
        return Antique(strength=0.6).apply_filter(result)



def filter2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(img_padded, (kh, kw))
    
    return np.einsum('ijkl,kl->ij', windows, kernel)

@dataclass
class Neon(ImageFilter):
    neon_strength: float = 1.5
    neon_blur: float = 0.8
    name = "neon"

    @staticmethod
    def _blur(image: np.ndarray, iterations: int) -> np.ndarray:
        #Размытие усреднением 3x3
        result = image.astype(np.float32)

        for _ in range(iterations):
            p = np.pad(result, 1, mode="edge")
            result = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0

        return result

    def apply_filter(self, img: Image) -> Image:
        if self.neon_strength < 0:
            raise ValueError("neon-strength должен быть >= 0")

        if self.neon_blur < 0:
            raise ValueError("neon-blur должен быть >= 0")

        gray = RGB2GrayScale().apply_filter(img).astype(
            np.float32
        )

        # Поиск границ разностями соседних пикселей
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        gy[:-1, :] = gray[1:, :] - gray[:-1, :]

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        edges = np.clip(magnitude / 255.0, 0.0, 1.0)

        # Два слоя свечения
        # neon-blur управляет шириной ореола.
        soft_iterations = max(0, round(4 * self.neon_blur))
        wide_iterations = max(0, round(10 * self.neon_blur))

        soft_glow = self._blur(edges, soft_iterations)
        wide_glow = self._blur(edges, wide_iterations)

        # neon-strength управляет яркостью свечения.
        glow = self.neon_strength * (
            edges * 70.0
            + soft_glow * 130.0
            + wide_glow * 180.0
        )

        result = img.astype(np.float32).copy()

        # BGR: синий + красный = фиолетовый
        result[:, :, 0] += glow
        result[:, :, 2] += glow

        return np.clip(result, 0, 255).astype(np.uint8)


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
