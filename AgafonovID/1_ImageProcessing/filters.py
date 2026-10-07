from abc import ABC, abstractmethod
import numpy as np


class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет фильтр к входному изображению."""
        pass

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> "ImageFilter":
        """Фабричный метод для создания объектов фильтров."""
        filter_type = filter_type.lower()

        if filter_type in ("grayscale", "gray"):
            return RGB2GrayScale()
        elif filter_type in ("antique", "sepia"):
            return AntiqueFilter()
        elif filter_type == "resize":
            width = kwargs.get("width")
            height = kwargs.get("height")
            if not width or not height:
                raise ValueError("Для фильтра resize требуются параметры --width и --height")
            return ResizeFilter(int(width), int(height))
        elif filter_type == "fade":
            alpha = float(kwargs.get("alpha", 0.6))
            return FadeFilter(alpha)
        elif filter_type in ("film", "infrared"):
            return FilmFilter()
        elif filter_type == "matte":
            return MatteFilter()
        elif filter_type == "scratches":
            noise_ratio = float(kwargs.get("noise_ratio", 0.015))
            num_lines = int(kwargs.get("num_lines", 12))
            return ScratchesNoiseFilter(noise_ratio, num_lines)
        elif filter_type == "neon":
            return NeonFilter()
        else:
            raise ValueError(f"Неизвестный тип фильтра: '{filter_type}'")


class ResizeFilter(ImageFilter):
    def __init__(self, target_width: int, target_height: int):
        self.target_width = target_width
        self.target_height = target_height

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        orig_height, orig_width = image.shape[:2]

        row_indices = (np.arange(self.target_height) * (orig_height / self.target_height)).astype(int)
        col_indices = (np.arange(self.target_width) * (orig_width / self.target_width)).astype(int)

        row_indices = np.clip(row_indices, 0, orig_height - 1)
        col_indices = np.clip(col_indices, 0, orig_width - 1)

        return image[np.ix_(row_indices, col_indices)]


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return np.clip(gray, 0, 255).astype(np.uint8)

class AntiqueFilter(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        r_new = 0.393 * r + 0.769 * g + 0.189 * b
        g_new = 0.349 * r + 0.686 * g + 0.168 * b
        b_new = 0.272 * r + 0.534 * g + 0.131 * b

        result = np.stack([b_new, g_new, r_new], axis=-1)
        return np.clip(result, 0, 255).astype(np.uint8)

class FadeFilter(ImageFilter):
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32)
        faded = img_float * self.alpha + (1.0 - self.alpha) * 255.0
        return np.clip(faded, 0, 255).astype(np.uint8)

class FilmFilter(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        r_new = np.clip(g * 1.6 + r * 0.4, 0, 255)
        g_new = np.clip(g * 0.3 + b * 0.2, 0, 255)
        b_new = np.clip(b * 0.8, 0, 255)

        result = np.stack([b_new, g_new, r_new], axis=-1)
        return result.astype(np.uint8)

class MatteFilter(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        dist = ((x - cx) ** 2) / ((cx * 0.85) ** 2) + ((y - cy) ** 2) / ((cy * 0.85) ** 2)
        mask = np.clip((dist - 0.7) / 0.3, 0, 1)

        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]

        result = image.astype(np.float32) * (1.0 - mask) + 255.0 * mask
        return np.clip(result, 0, 255).astype(np.uint8)

class ScratchesNoiseFilter(ImageFilter):
    def __init__(self, noise_ratio: float = 0.015, num_lines: int = 12):
        self.noise_ratio = noise_ratio
        self.num_lines = num_lines

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        res = image.copy()
        h, w = res.shape[:2]

        num_noise = int(self.noise_ratio * h * w)
        ys = np.random.randint(0, h, num_noise)
        xs = np.random.randint(0, w, num_noise)
        res[ys, xs] = 255

        for _ in range(self.num_lines):
            x0 = np.random.randint(0, w)
            y0 = np.random.randint(0, h // 2)
            length = np.random.randint(h // 6, h // 2)
            dx = np.random.randint(-5, 6)

            y_coords = np.clip(np.arange(y0, min(y0 + length, h)), 0, h - 1)
            x_coords = np.clip(x0 + ((y_coords - y0) * dx // length), 0, w - 1)

            scratch_color = np.random.randint(200, 256)
            res[y_coords, x_coords] = scratch_color

        return res

class NeonFilter(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img_gray = image.astype(np.float32)
        if len(img_gray.shape) == 3:
            img_gray = 0.114 * img_gray[:, :, 0] + 0.587 * img_gray[:, :, 1] + 0.299 * img_gray[:, :, 2]

        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        h, w = img_gray.shape
        padded = np.pad(img_gray, 1, mode='edge')
        
        gx = np.zeros_like(img_gray)
        gy = np.zeros_like(img_gray)

        for i in range(3):
            for j in range(3):
                gx += padded[i:i+h, j:j+w] * kernel_x[i, j]
                gy += padded[i:i+h, j:j+w] * kernel_y[i, j]

        magnitude = np.sqrt(gx**2 + gy**2)
        edges = np.clip(magnitude, 0, 255).astype(np.uint8)

        neon = np.zeros((*edges.shape, 3), dtype=np.uint8)
        neon[:, :, 0] = np.clip(edges * 1.2, 0, 255).astype(np.uint8)
        neon[:, :, 1] = np.clip(edges * 1.5, 0, 255).astype(np.uint8)
        neon[:, :, 2] = np.clip(edges * 0.4, 0, 255).astype(np.uint8)
        return neon