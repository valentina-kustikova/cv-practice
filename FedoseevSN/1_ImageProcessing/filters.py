from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):
    @staticmethod
    def get_filter(filter_name, **params):
        registry = {
            "resize": Resize,
            "gray": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "scratches": ScratchesNoise,
            "neon": Neon,
        }
        if filter_name not in registry:
            raise ValueError(f"Неизвестный фильтр: {filter_name}")
        return registry[filter_name](**params)

    @abstractmethod
    def apply_filter(self, image):
        pass


class Resize(ImageFilter):
    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Ширина и высота должны быть > 0")
        self.width = width
        self.height = height

    def apply_filter(self, image):
        h, w = image.shape[:2]
        y_idx = np.clip((np.arange(self.height) * h / self.height).astype(int), 0, h - 1)
        x_idx = np.clip((np.arange(self.width) * w / self.width).astype(int), 0, w - 1)
        return image[y_idx][:, x_idx]


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return np.clip(gray, 0, 255).astype(np.uint8)


class Antique(ImageFilter):
    def __init__(self, intensity=1.0):
        self.intensity = float(np.clip(intensity, 0.0, 1.0))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        sr = 0.393 * r + 0.769 * g + 0.189 * b
        sg = 0.349 * r + 0.686 * g + 0.168 * b
        sb = 0.272 * r + 0.534 * g + 0.131 * b

        gamma = 1.2
        sr = 255.0 * (np.clip(sr, 0, 255) / 255.0) ** gamma
        sg = 255.0 * (np.clip(sg, 0, 255) / 255.0) ** gamma
        sb = 255.0 * (np.clip(sb, 0, 255) / 255.0) ** gamma

        h, w = img.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - 0.5 * (dist / max_dist)

        sr *= vignette
        sg *= vignette
        sb *= vignette

        antique = np.stack([sb, sg, sr], axis=2)
        result = img * (1.0 - self.intensity) + antique * self.intensity
        return np.clip(result, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def __init__(self, strength=0.3):
        self.strength = float(np.clip(strength, 0.0, 1.0))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        faded = img + (255.0 - img) * self.strength
        mean = faded.mean(axis=2, keepdims=True)
        faded = mean + (faded - mean) * (1.0 - self.strength * 0.5)
        return np.clip(faded, 0, 255).astype(np.uint8)


class Film(ImageFilter):
    def __init__(self, grain=15.0, tint=0.6):
        self.grain = float(grain)
        self.tint = float(np.clip(tint, 0.0, 1.0))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        new_r = np.clip(g * 1.3, 0, 255)
        new_g = np.clip(r * 0.8, 0, 255)
        new_b = np.clip(b * 0.5, 0, 255)
        infrared = np.stack([new_b, new_g, new_r], axis=2)

        result = img * (1.0 - self.tint) + infrared * self.tint
        noise = np.random.normal(0.0, self.grain, result.shape)
        result = result + noise
        return np.clip(result, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def __init__(self, border=0.1, softness=0.05):
        self.border = float(np.clip(border, 0.0, 0.49))
        self.softness = float(max(softness, 1e-3))

    def apply_filter(self, image):
        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]

        rx = cx * (1.0 - self.border)
        ry = cy * (1.0 - self.border)
        dist = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)

        mask = np.clip((1.0 - dist) / self.softness + 0.5, 0.0, 1.0)[:, :, np.newaxis]

        white = np.full_like(image, 255, dtype=np.float32)
        result = image.astype(np.float32) * mask + white * (1.0 - mask)
        return np.clip(result, 0, 255).astype(np.uint8)


class ScratchesNoise(ImageFilter):
    def __init__(self, n_scratches=15, noise_level=20.0):
        self.n_scratches = int(n_scratches)
        self.noise_level = float(noise_level)

    def apply_filter(self, image):
        result = image.astype(np.float32)
        h, w = result.shape[:2]

        noise = np.random.normal(0.0, self.noise_level, result.shape)
        result = result + noise

        for _ in range(self.n_scratches):
            x_start = np.random.randint(0, w)
            y_start = np.random.randint(0, h)
            length = np.random.randint(h // 4, h)
            angle = np.pi / 2 + np.random.uniform(-0.3, 0.3)
            brightness = np.random.randint(180, 256)

            for t in range(length):
                x = int(x_start + t * np.cos(angle))
                y = int(y_start + t * np.sin(angle))
                if 0 <= x < w and 0 <= y < h:
                    result[y, x] = brightness

        return np.clip(result, 0, 255).astype(np.uint8)


class Neon(ImageFilter):
    def __init__(self, threshold=50.0, glow_intensity=0.8):
        self.threshold = float(threshold)
        self.glow_intensity = float(np.clip(glow_intensity, 0.0, 1.0))

    def apply_filter(self, image):
        img = image.astype(np.float32)

        gray = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

        gy, gx = np.gradient(gray)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        edges = np.clip(magnitude - self.threshold, 0.0, None)
        if edges.max() > 0:
            edges = edges / edges.max() * 255.0

        glow = edges.copy()
        for shift in range(1, 4):
            glow = np.maximum(glow, np.roll(edges, shift, axis=0))
            glow = np.maximum(glow, np.roll(edges, -shift, axis=0))
            glow = np.maximum(glow, np.roll(edges, shift, axis=1))
            glow = np.maximum(glow, np.roll(edges, -shift, axis=1))

        neon_r = glow
        neon_g = glow * 0.3
        neon_b = glow * 0.9
        neon_layer = np.stack([neon_b, neon_g, neon_r], axis=2)

        result = img * (1.0 - self.glow_intensity) + neon_layer * self.glow_intensity
        return np.clip(result, 0, 255).astype(np.uint8)