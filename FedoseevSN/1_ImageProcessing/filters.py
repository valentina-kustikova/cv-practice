from abc import ABC, abstractmethod
import numpy as np


class ImageFilter(ABC):
    @staticmethod
    def get_filter(filter_name, **params):
        filters_map = {
            "resize": Resize,
            "gray": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "scratches": ScratchesNoise,
            "neon": Neon,
        }
        if filter_name not in filters_map:
            raise ValueError(f"Фильтр '{filter_name}' не найден")
        return filters_map[filter_name](**params)

    @abstractmethod
    def apply_filter(self, image):
        pass


class Resize(ImageFilter):
    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Размеры должны быть положительными")
        self.w = int(width)
        self.h = int(height)

    def apply_filter(self, image):
        src_h, src_w = image.shape[:2]
        
        y_points = np.linspace(0, src_h - 1, self.h)
        x_points = np.linspace(0, src_w - 1, self.w)
        
        y0 = np.floor(y_points).astype(int)
        x0 = np.floor(x_points).astype(int)
        y1 = np.minimum(y0 + 1, src_h - 1)
        x1 = np.minimum(x0 + 1, src_w - 1)
        
        dy = (y_points - y0)
        dx = (x_points - x0)
        
        if image.ndim == 3:
            dy = dy[:, np.newaxis, np.newaxis]
            dx = dx[np.newaxis, :, np.newaxis]
        else:
            dy = dy[:, np.newaxis]
            dx = dx[np.newaxis, :]
            
        row0 = image[y0[:, None], x0] * (1 - dx) + image[y0[:, None], x1] * dx
        row1 = image[y1[:, None], x0] * (1 - dx) + image[y1[:, None], x1] * dx
        
        res = row0 * (1 - dy) + row1 * dy
        return np.clip(res, 0, 255).astype(np.uint8)


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        gray = image.astype(np.float32) @ weights
        return np.clip(gray, 0, 255).astype(np.uint8)


class Antique(ImageFilter):
    def __init__(self, intensity=0.85):
        self.alpha = max(0.0, min(1.0, float(intensity)))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        
        sepia_mat = np.array([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393]
        ], dtype=np.float32)
        
        sepia_img = img @ sepia_mat.T
        
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w / 2.0, h / 2.0
        
        dist = np.sqrt(((x - center_x) / center_x) ** 2 + ((y - center_y) / center_y) ** 2)
        mask = np.clip(1.0 - 0.5 * dist, 0.0, 1.0)
        
        if img.ndim == 3:
            mask = mask[..., np.newaxis]
            
        res = img * (1.0 - self.alpha) + (sepia_img * mask) * self.alpha
        return np.clip(res, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def __init__(self, strength=0.35):
        self.fade = max(0.0, min(1.0, float(strength)))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        
        mixed = img * (1.0 - self.fade) + 255.0 * self.fade
        mean_val = mixed.mean(axis=2, keepdims=True)
        
        res = mean_val + (mixed - mean_val) * (1.0 - self.fade * 0.5)
        return np.clip(res, 0, 255).astype(np.uint8)


class Film(ImageFilter):
    def __init__(self, grain=15.0, tint=0.4):
        self.grain = float(grain)
        self.tint = max(0.0, min(1.0, float(tint)))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        
        color_mat = np.array([
            [0.2, 0.7, 0.1],
            [0.4, 0.4, 0.2],
            [0.9, 0.05, 0.05]
        ], dtype=np.float32)
        
        infrared = img @ color_mat.T
        res = img * (1.0 - self.tint) + infrared * self.tint
        
        if self.grain > 0:
            noise = np.random.normal(0, self.grain, res.shape)
            res += noise
            
        return np.clip(res, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def __init__(self, radius=0.75, softness=0.2):
        self.rad = max(0.1, min(1.0, float(radius)))
        self.soft = max(0.01, float(softness))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        
        d = np.sqrt(((x - cx) / (cx * self.rad)) ** 2 + ((y - cy) / (cy * self.rad)) ** 2)
        
        weight = np.clip((d - 1.0 + self.soft) / self.soft, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        
        if img.ndim == 3:
            weight = weight[..., np.newaxis]
            
        res = img * (1.0 - weight) + 255.0 * weight
        return np.clip(res, 0, 255).astype(np.uint8)


class ScratchesNoise(ImageFilter):
    def __init__(self, n_scratches=20, noise_level=15.0):
        self.count = int(n_scratches)
        self.sigma = float(noise_level)
        self.rng = np.random.default_rng()

    def apply_filter(self, image):
        res = image.astype(np.float32)
        h, w = res.shape[:2]

        res += self.rng.normal(0.0, self.sigma, res.shape)
        
        scratch_mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(self.count):
            start_x = self.rng.integers(0, w)
            start_y = self.rng.integers(0, h // 4)
            length = self.rng.integers(h // 2, h)
            
            end_x = int(start_x + self.rng.integers(-10, 10))
            end_y = int(start_y + length)
            
            pts = max(abs(end_x - start_x), abs(end_y - start_y), 1)
            xs = np.linspace(start_x, end_x, pts).round().astype(int)
            ys = np.linspace(start_y, end_y, pts).round().astype(int)
            
            mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            scratch_mask[ys[mask], xs[mask]] = 1.0

        if image.ndim == 3:
            scratch_mask = scratch_mask[..., np.newaxis]
            
        res += scratch_mask * 50.0
        return np.clip(res, 0, 255).astype(np.uint8)


class Neon(ImageFilter):
    def __init__(self, threshold=40, glow=1.5):
        self.thr = float(threshold)
        self.glow = float(glow)

    def apply_filter(self, image):
        img = image.astype(np.float32)
        
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        gray = img @ weights
        
        padded = np.pad(gray, 1, mode='edge')
        
        gx = (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) - \
             (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
             
        gy = (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) - \
             (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])

        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = np.clip(mag * self.glow, 0, 255)
        
        edges = (mag > self.thr).astype(np.float32)
        
        if img.ndim == 3:
            glow_layer = np.zeros_like(img)
            glow_layer[..., 0] = edges * mag * 0.2
            glow_layer[..., 1] = edges * mag * 0.9
            glow_layer[..., 2] = edges * mag * 1.0
            
            res = img * 0.25 + glow_layer
        else:
            res = edges * mag
            
        return np.clip(res, 0, 255).astype(np.uint8)