from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):

    @staticmethod
    def get_filter(name, param=None):
        if name == "gray":
            return GrayScale()
        if name == "resize":
            return Resize(param)
        if name == "antique":
            return Antique(param)
        if name == "fade_color":
            return FadeColor(param)
        if name == "film_effect":
            return FilmEffect(param)
        if name == "matte":     
            return Matte(param)
        if name == "old_photo":
            return OldPhoto(param)
        if name == "neon":
            return Neon(param)
        
        raise ValueError(f"Неизвестный фильтр: {name}")

    @abstractmethod
    def apply_filter(self, image):
        pass


class GrayScale(ImageFilter):

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return np.clip(gray, 0, 255).astype(np.uint8)

class Resize(ImageFilter):
    def __init__(self, scale=None):
        if scale is None:
            scale = 0.5
        if scale <= 0:
            raise ValueError("Масштаб должен быть больше 0")
        self.scale = scale
    def apply_filter(self, image):
        height, width = image.shape[:2]

        new_height = int(height * self.scale)
        new_width = int(width * self.scale)

        rows = (np.arange(new_height) / self.scale).astype(int)
        cols = (np.arange(new_width) / self.scale).astype(int)

        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)

        return image[rows][:, cols]

class Antique(ImageFilter):

    def __init__(self, param=None):
        if param is None:
            param = 1.0
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_b = 0.272 * r + 0.534 * g + 0.131 * b

        sepia = np.stack([new_b, new_g, new_r], axis=2)

        result = (1 - self.param) * img + self.param * sepia

        return np.clip(result, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def __init__(self, param=None):
        if param is None:
            param = 0.5
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        faded = img * (1 - self.param) + 255 * self.param

        return np.clip(faded, 0, 255).astype(np.uint8)


class FilmEffect(ImageFilter):

    def __init__(self, param=None):
        if param is None:
            param = 1.0
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        infrared = np.stack([b, r, g], axis=2)

        result = (1 - self.param) * img + self.param * infrared

        return np.clip(result, 0, 255).astype(np.uint8)

    
class Matte(ImageFilter):

    def __init__(self, param=None):
        if param is None:
            param = 0.6
        if not (0 <= param < 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1)")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)
        height, width = img.shape[:2]

        cx = width / 2
        cy = height / 2

        x = np.arange(width)
        y = np.arange(height)

        dx = (x[None, :] - cx) / (width / 2)
        dy = (y[:, None] - cy) / (height / 2)

        d = np.sqrt(dx ** 2 + dy ** 2)

        mask = (1 - d) / (1 - self.param)
        mask = np.clip(mask, 0, 1)
        mask = mask[:, :, None]

        result = img * mask + 255 * (1 - mask)

        return np.clip(result, 0, 255).astype(np.uint8)

class OldPhoto(ImageFilter):

    def __init__(self, param=None):
        if param is None:
            param = 0.5
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)
        height, width = img.shape[:2]

        sigma = 25 * self.param
        noise = np.random.normal(0, sigma, (height, width, 1))
        img = img + noise

        scratches = int(20 * self.param)
        for _ in range(scratches):
            x = np.random.randint(0, width)
            y1 = np.random.randint(0, height // 2)
            y2 = np.random.randint(height // 2, height)
            img[y1:y2, x] = img[y1:y2, x] + 80

        dust = int(300 * self.param)
        ys = np.random.randint(0, height, dust)
        xs = np.random.randint(0, width, dust)
        img[ys, xs] = 30

        return np.clip(img, 0, 255).astype(np.uint8)

class Neon(ImageFilter):

    def __init__(self, param=None):
        if param is None:
            param = 30
        if not (0 < param < 255):
            raise ValueError("Порог должен быть в диапазоне (0, 255)")
        self.threshold = param

    def apply_filter(self, image):
        img = image.astype(np.float32)
        brightness = img.mean(axis=2)

        dx = np.zeros_like(brightness)
        dy = np.zeros_like(brightness)
        dx[:, :-1] = np.abs(brightness[:, 1:] - brightness[:, :-1])
        dy[:-1, :] = np.abs(brightness[1:, :] - brightness[:-1, :])

        mask = (dx + dy) > self.threshold

        result = np.zeros_like(img)
        result[mask] = img[mask] * 2

        return np.clip(result, 0, 255).astype(np.uint8)

