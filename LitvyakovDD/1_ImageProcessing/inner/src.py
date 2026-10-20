from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


SCRATCHES_PATH = Path(__file__).with_name("Scratches.jpg")


def clip_image(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def to_gray(image):
    image = image.astype(np.float32)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    return 0.114 * blue + 0.587 * green + 0.299 * red


def check_image(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Нужно цветное изображение.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("У изображения неправильный размер.")


def center_crop(image, height, width):
    top = (image.shape[0] - height) // 2
    left = (image.shape[1] - width) // 2
    return image[top:top + height, left:left + width]


class ImageFilter(ABC):
    @staticmethod
    def get_filter(filter_name, **kwargs):
        filters = {
            "resize": Resize,
            "grayscale": RGB2GrayScale,
            "antique": Antique,
            "fade": Fade,
            "film": Film,
            "matte": Matte,
            "scratches": Scratches,
            "neon": Neon,
        }
        try:
            return filters[filter_name](**kwargs)
        except KeyError as exc:
            raise ValueError(f"Неизвестный фильтр: {filter_name}") from exc

    @abstractmethod
    def apply_filter(self, image, **kwargs):
        raise NotImplementedError


class Resize(ImageFilter):
    def __init__(self, width=None, height=None):
        self.width = width
        self.height = height

    def apply_filter(self, image, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        if width is None or height is None:
            raise ValueError("Для resize нужно указать ширину и высоту.")
        if width <= 0 or height <= 0:
            raise ValueError("Ширина и высота должны быть больше нуля.")

        image_height, image_width = image.shape[:2]
        x_indices = (np.arange(width) * image_width / width).astype(int)
        y_indices = (np.arange(height) * image_height / height).astype(int)
        return image[y_indices[:, None], x_indices]


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image, **kwargs):
        return clip_image(to_gray(image))


class Antique(ImageFilter):
    def __init__(self, strength=1.0):
        self.strength = strength

    def apply_filter(self, image, strength=None):
        strength = self.strength if strength is None else strength
        image = image.astype(np.float32)
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]
        sepia = np.stack((
            0.272 * red + 0.534 * green + 0.131 * blue,
            0.349 * red + 0.686 * green + 0.168 * blue,
            0.393 * red + 0.769 * green + 0.189 * blue,
        ), axis=2)
        return clip_image(image * (1 - strength) + sepia * strength)


class Fade(ImageFilter):
    def __init__(self, strength=1.0):
        self.strength = strength

    def apply_filter(self, image, strength=None):
        strength = self.strength if strength is None else strength
        original = image.astype(np.float32)
        gray = to_gray(image)[:, :, None]
        faded = original * (1 - 0.55 * strength) + gray * (0.55 * strength)
        faded = faded * (1 - 0.18 * strength) + 245 * (0.18 * strength)
        return clip_image(faded)


class Film(ImageFilter):
    def __init__(self, strength=1.0):
        self.strength = strength

    def apply_filter(self, image, strength=None):
        strength = self.strength if strength is None else strength
        image = image.astype(np.float32)
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]
        infrared = np.stack((
            0.65 * blue + 0.35 * red,
            0.75 * red + 0.25 * green,
            1.35 * green - 0.25 * blue + 20,
        ), axis=2)
        return clip_image(image * (1 - strength) + infrared * strength)


class Matte(ImageFilter):
    def __init__(self, matte_radius_x=0.8, matte_radius_y=0.8, matte_softness=0.2):
        self.matte_radius_x = matte_radius_x
        self.matte_radius_y = matte_radius_y
        self.matte_softness = matte_softness

    def apply_filter(
        self, image, matte_radius_x=None, matte_radius_y=None, matte_softness=None
    ):
        matte_radius_x = (
            self.matte_radius_x if matte_radius_x is None else matte_radius_x
        )
        matte_radius_y = (
            self.matte_radius_y if matte_radius_y is None else matte_radius_y
        )
        matte_softness = (
            self.matte_softness if matte_softness is None else matte_softness
        )
        if matte_radius_x <= 0:
            raise ValueError("matte_radius_x must be positive")
        if matte_radius_y <= 0:
            raise ValueError("matte_radius_y must be positive")
        if matte_softness < 0:
            raise ValueError("matte_softness must be non-negative")

        height, width = image.shape[:2]
        y_center, x_center = height / 2, width / 2
        y, x = np.ogrid[:height, :width]
        distance = (
            (x - x_center) ** 2 / (x_center * matte_radius_x) ** 2
            + (y - y_center) ** 2 / (y_center * matte_radius_y) ** 2
        )

        if matte_softness == 0:
            weight = (distance > 1).astype(np.float32)
        else:
            weight = np.clip(
                (np.sqrt(distance) - 1) / matte_softness, 0, 1
            ).astype(np.float32)
        weight = weight[:, :, None]

        result = image.astype(np.float32) * (1 - weight) + 255 * weight
        return clip_image(result)


class Scratches(ImageFilter):
    def __init__(self, strength=0.3):
        self.strength = strength

    def apply_filter(self, image, strength=None):
        strength = self.strength if strength is None else strength
        texture_data = np.fromfile(SCRATCHES_PATH, dtype=np.uint8)
        texture = cv2.imdecode(texture_data, cv2.IMREAD_COLOR)
        if texture is None:
            raise ValueError(f"Не удалось открыть текстуру царапин: {SCRATCHES_PATH}")

        height, width = image.shape[:2]
        texture_height, texture_width = texture.shape[:2]
        if texture_height < height or texture_width < width:
            texture = cv2.resize(texture, (width, height))
        else:
            texture = center_crop(texture, height, width)

        result = (
            image.astype(np.float32) * (1 - strength)
            + texture.astype(np.float32) * strength
        )
        return clip_image(result)


class Neon(ImageFilter):
    def __init__(self, neon_gain=9.0, neon_brightness=0.4):
        self.neon_gain = neon_gain
        self.neon_brightness = neon_brightness

    def apply_filter(self, image, neon_gain=None, neon_brightness=None):
        neon_gain = self.neon_gain if neon_gain is None else neon_gain
        neon_brightness = (
            self.neon_brightness
            if neon_brightness is None
            else neon_brightness
        )
        if neon_gain <= 0:
            raise ValueError("neon_gain must be positive")
        if not 0 <= neon_brightness <= 1:
            raise ValueError("neon_brightness must be between 0 and 1")

        gray = RGB2GrayScale().apply_filter(image).astype(np.float32)
        horizontal_gradient = np.zeros_like(gray)
        vertical_gradient = np.zeros_like(gray)
        horizontal_gradient[:, :-1] = gray[:, 1:] - gray[:, :-1]
        vertical_gradient[:-1, :] = gray[1:, :] - gray[:-1, :]

        magnitude = np.sqrt(
            horizontal_gradient**2 + vertical_gradient**2
        )
        magnitude = np.clip(magnitude * neon_gain, 0, 255)
        magnitude = cv2.dilate(
            magnitude, np.ones((3, 3), dtype=np.uint8)
        )
        result = image.astype(np.float32) * neon_brightness
        result[:, :, 0] += magnitude
        result[:, :, 2] += magnitude
        return clip_image(result)


FILTERS = {
    "resize": Resize().apply_filter,
    "grayscale": RGB2GrayScale().apply_filter,
    "antique": Antique().apply_filter,
    "fade": Fade().apply_filter,
    "film": Film().apply_filter,
    "matte": Matte().apply_filter,
    "scratches": Scratches().apply_filter,
    "neon": Neon().apply_filter,
}
