import numpy as np
from abc import ABC, abstractmethod


class ImageFilter(ABC):
    @staticmethod
    def get_filter(filter_name, **kwargs):
        filters = {
            'resize': Resize,
            'grayscale': Grayscale,
            'antique': Antique,
            'fade': Fade,
            'infrared': Infrared,
            'matte': Matte,
            'noise': Noise,
            'neon': Neon
        }
        if filter_name not in filters:
            raise ValueError(f"Filter '{filter_name}' not found.")
        return filters[filter_name](**kwargs)

    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        pass


class Resize(ImageFilter):
    def __init__(self, width=800, height=600):
        self.width = int(width)
        self.height = int(height)

    def apply_filter(self, image):
        h, w, c = image.shape
        x_ratio = w / self.width
        y_ratio = h / self.height
        
        x_idx = (np.arange(self.width) * x_ratio).astype(int)
        y_idx = (np.arange(self.height) * y_ratio).astype(int)
        
        return image[np.ix_(y_idx, x_idx)]


class Grayscale(ImageFilter):
    def apply_filter(self, image):
        weights = np.array([0.114, 0.587, 0.299])
        gray = np.dot(image[..., :3], weights).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


class Antique(ImageFilter):
    def apply_filter(self, image):
        matrix = np.array([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393]
        ])
        sepia = np.dot(image, matrix.T)
        return np.clip(sepia, 0, 255).astype(np.uint8)


class Fade(ImageFilter):
    def __init__(self, fade_factor=0.6):
        self.fade_factor = float(fade_factor)

    def apply_filter(self, image):
        weights = np.array([0.114, 0.587, 0.299])
        gray_1d = np.dot(image, weights)
        gray_3d = np.stack([gray_1d] * 3, axis=-1)
        
        faded = image * (1 - self.fade_factor) + gray_3d * self.fade_factor
        return np.clip(faded, 0, 255).astype(np.uint8)


class Infrared(ImageFilter):
    def apply_filter(self, image):
        ir_img = np.zeros_like(image, dtype=np.float32)
        ir_img[:, :, 0] = image[:, :, 0] * 0.3
        ir_img[:, :, 1] = image[:, :, 1] * 0.3
        ir_img[:, :, 2] = 255 - image[:, :, 2] * 0.8
        return np.clip(ir_img, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def apply_filter(self, image):
        h, w = image.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        dist = ((X - center_x) ** 2) / (w / 2) ** 2 + ((Y - center_y) ** 2) / (h / 2) ** 2
        
        mask = np.clip(dist, 0, 1)
        mask_3d = np.stack([mask] * 3, axis=-1)
        
        white = np.ones_like(image) * 255
        matted = image * (1 - mask_3d) + white * mask_3d
        return matted.astype(np.uint8)


class Noise(ImageFilter):
    def apply_filter(self, image):
        h, w, c = image.shape
        noisy = image.astype(np.float32)
        
        gauss = np.random.normal(0, 15, (h, w, c))
        noisy += gauss
        
        num_scratches = np.random.randint(10, 30)
        for _ in range(num_scratches):
            x = np.random.randint(0, w)
            y1 = np.random.randint(0, h // 2)
            y2 = np.random.randint(h // 2, h)
            noisy[y1:y2, x:x + 1] = 200 + np.random.randint(0, 55)
            
        return np.clip(noisy, 0, 255).astype(np.uint8)


class Neon(ImageFilter):
    def apply_filter(self, image):
        gray = np.dot(image, [0.114, 0.587, 0.299]).astype(np.float32)
        
        diff_x = np.zeros_like(gray)
        diff_y = np.zeros_like(gray)
        
        diff_x[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
        diff_y[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
        
        edges = np.clip(diff_x + diff_y, 0, 255)
        
        neon = np.zeros_like(image)
        neon[:, :, 0] = edges * 1.0
        neon[:, :, 1] = edges * 0.8
        neon[:, :, 2] = edges * 0.1
        
        return np.clip(neon, 0, 255).astype(np.uint8)


