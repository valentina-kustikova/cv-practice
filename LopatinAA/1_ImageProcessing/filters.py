from abc import ABC, abstractmethod

import cv2
import numpy as np


class ImageFilter(ABC):
    @staticmethod
    def get_filter(filter_type, **params):
        filters = {
            'resize': Resize,
            'grayscale': RGB2GrayScale,
            'antique': Antique,
            'fadecolor': FadeColor,
            'tape': Tape,
            'matte': Matte,
            'noise': Noise,
            'neon': Neon,
        }
        key = filter_type.lower()
        if key not in filters:
            raise ValueError(f'Неизвестный фильтр: {filter_type}')
        return filters[key](**params)

    @abstractmethod
    def apply_filter(self, image):
        pass

class Resize(ImageFilter):
    def __init__(self, width=None, height=None):
        if width is None or height is None:
            raise ValueError('Требуется задать width и height')
        self.width = width
        self.height = height

    def apply_filter(self, image):
        h, w = image.shape[:2]
        nh, nw = self.height, self.width
        yi = np.clip(np.round(np.linspace(0, h - 1, nh)).astype(int), 0, h - 1)
        xi = np.clip(np.round(np.linspace(0, w - 1, nw)).astype(int), 0, w - 1)
        return image[yi][:, xi]

class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return np.clip(gray, 0, 255).astype(np.uint8)

class Antique(ImageFilter):
    def __init__(self, intensity=0.8):
        self.intensity = intensity

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_b = 0.272 * r + 0.534 * g + 0.131 * b
        sepia = np.stack([new_b, new_g, new_r], axis=-1)

        out = self.intensity * sepia + (1 - self.intensity) * img

        return np.clip(out, 0, 255).astype(np.uint8)

class FadeColor(ImageFilter):
    def __init__(self, strength=0.5):
        self.strength = strength

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = (0.2126 * r + 0.7152 * g + 0.0722 * b)[:, :, None]

        out = img * (1 - self.strength) + gray * self.strength
        out = out * (1 - self.strength * 0.3) + 255 * (self.strength * 0.3)

        return np.clip(out, 0, 255).astype(np.uint8)

class Tape(ImageFilter):
    def __init__(self, grain=0.1):
        self.grain = grain

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        new_r = 0.2 * r + 1.3 * g - 0.5 * b
        new_g = 0.0 * r + 0.4 * g + 0.4 * b
        new_b = 0.0 * r + 0.0 * g + 1.0 * b
        out = np.stack([new_b, new_g, new_r], axis=-1)

        if self.grain > 0:
            noise = np.random.normal(0, self.grain * 255, out.shape)
            out = out + noise

        return np.clip(out, 0, 255).astype(np.uint8)

class Matte(ImageFilter):
    def __init__(self, border=0.05, feather=0.05, color=(255, 255, 255)):
        self.border = border
        self.feather = feather
        self.color = color

    def apply_filter(self, image):
        img = image.astype(np.float32)
        h, w = img.shape[:2]

        cy, cx = h / 2, w / 2
        ry = max(cy * (1 - self.border), 1.0)
        rx = max(cx * (1 - self.border), 1.0)

        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)

        f = max(self.feather, 1e-6)
        mask = np.clip((1 - d) / f, 0, 1)[:, :, None]

        fill = np.zeros_like(img)
        fill[:, :, 0] = self.color[0]
        fill[:, :, 1] = self.color[1]
        fill[:, :, 2] = self.color[2]

        out = img * mask + fill * (1 - mask)
        return np.clip(out, 0, 255).astype(np.uint8)

class Noise(ImageFilter):
    def __init__(self, scratch_count=7, noise_level=0.05):
        self.scratch_count = scratch_count
        self.noise_level = noise_level

    def apply_filter(self, image):
        img = image.copy()
        h, w = img.shape[:2]
        rng = np.random.default_rng()

        for _ in range(self.scratch_count):
            x = int(rng.integers(0, w))
            y0 = int(rng.integers(0, max(h // 2, 1)))
            y1 = int(rng.integers(max(h // 2, 1), h))
            color = 255 if rng.random() < 0.5 else 0
            img[y0:y1, x] = color

        if self.noise_level > 0:
            mask = rng.random((h, w)) < self.noise_level
            salt = rng.random((h, w)) < 0.5
            img[mask & salt] = 255
            img[mask & ~salt] = 0

        return img

class Neon(ImageFilter):
    def __init__(self, threshold=50, glow=0.8, color=(255, 255, 50)):
        self.threshold = threshold
        self.glow = glow
        self.color = color

    @staticmethod
    def _blur3(image):
        return (
            image
            + np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0)
            + np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1)
            + np.roll(np.roll(image, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(image, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(image, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(image, -1, axis=0), -1, axis=1)
        ) / 9.0

    def apply_filter(self, image):
        img = image.astype(np.uint8)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)

        mask = (mag > self.threshold).astype(np.float32)
        glow_mask = mask

        blur_passes = 6
        for _ in range(blur_passes):
            glow_mask = self._blur3(glow_mask)

        glow_mask[:blur_passes, :] = 0
        glow_mask[-blur_passes:, :] = 0
        glow_mask[:, :blur_passes] = 0
        glow_mask[:, -blur_passes:] = 0

        cb, cg, cr = self.color

        edges = np.zeros_like(img, dtype=np.float32)
        edges[:, :, 0] = mask * cb
        edges[:, :, 1] = mask * cg
        edges[:, :, 2] = mask * cr

        glow_layer = np.zeros_like(img, dtype=np.float32)
        glow_layer[:, :, 0] = glow_mask * cb * 0.8
        glow_layer[:, :, 1] = glow_mask * cg * 0.8
        glow_layer[:, :, 2] = glow_mask * cr * 0.8

        out = edges + glow_layer * self.glow
        return np.clip(out, 0, 255).astype(np.uint8)