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

        noise = np.random.normal(0, 0.05 * self.intensity * 255, img.shape)
        img = img + noise

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
    def __init__(self, scratch_count=2, noise_level=0.04, intensity=0.8):
        self.scratch_count = max(0, scratch_count)
        self.noise_level = noise_level
        self.intensity = intensity

    @staticmethod
    def _smooth(arr, passes):
        for _ in range(passes):
            arr = (
                arr
                + np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
                + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1)
            ) / 5.0
        return arr

    def apply_filter(self, image):
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        rng = np.random.default_rng()

        # sepia
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_b = 0.272 * r + 0.534 * g + 0.131 * b
        sepia = np.stack([new_b, new_g, new_r], axis=-1)
        out = self.intensity * sepia + (1 - self.intensity) * img

        # paper texture
        texture = np.zeros((h, w), dtype=np.float32)
        for passes, weight in [(2, 0.4), (6, 0.4), (15, 0.2)]:
            layer = rng.random((h, w)).astype(np.float32)
            layer = self._smooth(layer, passes)
            layer = (layer - layer.min()) / (layer.max() - layer.min() + 1e-6)
            texture += weight * layer
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
        out = out * (0.85 + 0.15 * texture)[..., None]

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        # scratches
        fold_map = np.zeros((h, w), dtype=np.float32)
        diag = np.hypot(h, w)
        for _ in range(self.scratch_count):
            cy = rng.uniform(0, h)
            cx = rng.uniform(0, w)
            angle = rng.uniform(-1.0, 1.0)
            width = rng.uniform(3.0, 8.0)
            length = rng.uniform(0.6, 1.0)

            dist_signed = np.cos(angle) * (yy - cy) - np.sin(angle) * (xx - cx)
            dist = np.abs(dist_signed)

            line = np.exp(-(dist ** 2) / (2 * width ** 2))

            along = np.sin(angle) * (yy - cy) + np.cos(angle) * (xx - cx)
            half_len = 0.5 * length * diag
            line *= (np.abs(along) < half_len).astype(np.float32)

            side = np.sign(dist_signed)
            fold = line * side * rng.uniform(0.15, 0.28)
            fold_map += fold
        fold_map = np.clip(fold_map, -0.4, 0.4)

        gray_neutral = np.full_like(out, 128.0)
        alpha_fold = np.abs(fold_map)[..., None] * 0.85
        target = gray_neutral + fold_map[..., None] * 90
        out = out * (1 - alpha_fold) + target * alpha_fold

        # noise
        out = out + rng.normal(0, self.noise_level * 255, out.shape)

        # vignette
        cy_v, cx_v = h / 2, w / 2
        r2 = ((yy - cy_v) / cy_v) ** 2 + ((xx - cx_v) / cx_v) ** 2
        vignette = 1 - 0.35 * np.clip(r2, 0, 1)
        out = out * vignette[..., None]

        return np.clip(out, 0, 255).astype(np.uint8)

class Neon(ImageFilter):
    def __init__(self, threshold=40, glow=0.8, color=(255, 255, 50), glow_width=4):
        self.threshold = threshold
        self.glow = glow
        self.color = color
        self.glow_width = glow_width

    @staticmethod
    def _dilate(mask):
        out = mask.copy()
        out[1:, :] = np.maximum(out[1:, :], mask[:-1, :])
        out[:-1, :] = np.maximum(out[:-1, :], mask[1:, :])
        out[:, 1:] = np.maximum(out[:, 1:], mask[:, :-1])
        out[:, :-1] = np.maximum(out[:, :-1], mask[:, 1:])
        return out

    def _glow_by_dilation(self, mask, sigma):
        glow = mask * 0.4
        current = mask
        radius = max(1, int(sigma * 3))
        for r in range(1, radius + 1):
            current = self._dilate(current)
            weight = np.exp(-(r * r) / (2 * sigma * sigma)) * 0.4
            glow = np.maximum(glow, current * weight)
        return glow

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)

        mask = (mag > self.threshold).astype(np.float32)

        sigma = max(1.0, float(self.glow_width))
        glow_mask = self._glow_by_dilation(mask, sigma)

        cb, cg, cr = self.color
        base = img * 0.65

        edges = np.zeros_like(img)
        edges[:, :, 0] = mask * cb
        edges[:, :, 1] = mask * cg
        edges[:, :, 2] = mask * cr

        glow_layer = np.zeros_like(img)
        glow_layer[:, :, 0] = glow_mask * cb
        glow_layer[:, :, 1] = glow_mask * cg
        glow_layer[:, :, 2] = glow_mask * cr

        out = base + edges + glow_layer * self.glow
        return np.clip(out, 0, 255).astype(np.uint8)