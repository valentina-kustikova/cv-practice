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

#     def apply_filter(self, image, neon_gain=None, neon_brightness=None):
#         neon_gain = self.neon_gain if neon_gain is None else neon_gain
#         neon_brightness = (
#             self.neon_brightness
#             if neon_brightness is None
#             else neon_brightness
#         )
#         if neon_gain <= 0:
#             raise ValueError("neon_gain must be positive")
#         if not 0 <= neon_brightness <= 1:
#             raise ValueError("neon_brightness must be between 0 and 1")

#         gray = RGB2GrayScale().apply_filter(image).astype(np.float32)
#         horizontal_gradient = np.zeros_like(gray)
#         vertical_gradient = np.zeros_like(gray)
#         horizontal_gradient[:, :-1] = gray[:, 1:] - gray[:, :-1]
#         vertical_gradient[:-1, :] = gray[1:, :] - gray[:-1, :]

#         magnitude = np.sqrt(
#             horizontal_gradient**2 + vertical_gradient**2
#         )
#         magnitude = np.clip(magnitude * neon_gain, 0, 255)
#         magnitude = cv2.dilate(
#             magnitude, np.ones((5, 5), dtype=np.uint8)
#         )
#         result = image.astype(np.float32) * neon_brightness
#         result[:, :, 0] += magnitude
#         result[:, :, 2] += magnitude
#         return clip_image(result)

    # @staticmethod
    # def _blur(image: np.ndarray, iterations: int) -> np.ndarray:
    #     result = image.astype(np.float32)

    #     for _ in range(iterations):
    #         p = np.pad(result, 1, mode="edge")
    #         result = (
    #             p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
    #             p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
    #             p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    #         ) / 9.0

    #     return result

    # def apply_filter(self, img: Image) -> Image:
    #     if self.neon_gain < 0:
    #         raise ValueError("neon-strength должен быть >= 0")

    #     if self.neon_brightness < 0:
    #         raise ValueError("neon-blur должен быть >= 0")

    #     gray = RGB2GrayScale().apply_filter(img).astype(
    #         np.float32
    #     )

    #     gx = np.zeros_like(gray)
    #     gy = np.zeros_like(gray)

    #     gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
    #     gy[:-1, :] = gray[1:, :] - gray[:-1, :]

    #     magnitude = np.sqrt(gx ** 2 + gy ** 2)
    #     edges = np.clip(magnitude / 255.0, 0.0, 1.0)

    #     soft_iterations = max(0, round(4 * self.neon_brightness))
    #     wide_iterations = max(0, round(10 * self.neon_brightness))

    #     soft_glow = self._blur(edges, soft_iterations)
    #     wide_glow = self._blur(edges, wide_iterations)

    #     glow = self.neon_gain * (
    #         edges * 70.0
    #         + soft_glow * 130.0
    #         + wide_glow * 180.0
    #     )

    #     result = img.astype(np.float32).copy()

    #     result[:, :, 0] = result[:, :, 0] * 0.8 + glow
    #     result[:, :, 2] = result[:, :, 2] * 0.8 + glow

    #     return np.clip(result, 0, 255).astype(np.uint8)

    # def apply_filter(self, img, **kwargs):
    #     gain = kwargs.get("neon_gain", 5.0)
    #     if gain <= 0:
    #         raise ValueError("neon_gain must be positive")

    #     gray = RGB2GrayScale().apply_filter(img).astype(np.float32)

    #     gx = np.zeros_like(gray)
    #     gy = np.zeros_like(gray)

    #     gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
    #     gy[:-1, :] = gray[1:, :] - gray[:-1, :]

    #     magnitude = np.sqrt(gx**2 + gy**2)
    #     magnitude = np.clip(magnitude * gain, 0, 255).astype(np.uint8)

    #     #res = np.zeros_like(img)
    #     res = img.astype(np.float32)
    #     res[:, :, 0] += magnitude
    #     res[:, :, 1] += magnitude
    #     return np.clip(res, 0, 255).astype(np.uint8)

    
class Neon(ImageFilter):
    def __init__(self, neon_gain=40, neon_brightness=0.8, color=(255, 10, 149), neon_brightness_width=10):
        self.neon_gain = neon_gain
        self.neon_brightness = neon_brightness
        self.color = color
        self.neon_brightness_width = neon_brightness_width

    @staticmethod
    def _dilate(mask):
        out = mask.copy()
        out[1:, :] = np.maximum(out[1:, :], mask[:-1, :])
        out[:-1, :] = np.maximum(out[:-1, :], mask[1:, :])
        out[:, 1:] = np.maximum(out[:, 1:], mask[:, :-1])
        out[:, :-1] = np.maximum(out[:, :-1], mask[:, 1:])
        return out

    def _neon_brightness_by_dilation(self, mask, sigma):
        neon_brightness = mask * 0.4
        current = mask
        radius = max(1, int(sigma * 3))
        for r in range(1, radius + 1):
            current = self._dilate(current)
            weight = np.exp(-(r * r) / (2 * sigma * sigma)) * 0.4
            neon_brightness = np.maximum(neon_brightness, current * weight)
        return neon_brightness

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        
        gray = RGB2GrayScale().apply_filter(img).astype(np.float32)


        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        mask = (mag > self.neon_gain).astype(np.float32)
        for time in range(1,2):
            mask = self._dilate(mask)
        sigma = max(1.0, float(self.neon_brightness_width))
        neon_brightness_mask = self._neon_brightness_by_dilation(mask, sigma)

        cb, cg, cr = self.color
        base = img * 0.85

        edges = np.zeros_like(img)
        edges[:, :, 0] = mask * cb
        edges[:, :, 1] = mask * cg
        edges[:, :, 2] = mask * cr

        neon_brightness_layer = np.zeros_like(img)
        neon_brightness_layer[:, :, 0] = neon_brightness_mask * cb
        neon_brightness_layer[:, :, 1] = neon_brightness_mask * cg
        neon_brightness_layer[:, :, 2] = neon_brightness_mask * cr

        out = base + edges + neon_brightness_layer * self.neon_brightness
        return np.clip(out, 0, 255).astype(np.uint8)

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
