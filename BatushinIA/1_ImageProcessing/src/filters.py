# src/filters.py
from abc import ABC, abstractmethod
import numpy as np


class ImageFilter(ABC):
    @staticmethod
    def get_filter(name: str, **kwargs) -> "ImageFilter":
        """
        Фабричный метод: по строковому имени создаёт конкретный фильтр.
        """
        registry = {
            "resize": Resize,
            "gray": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "scratches": Scratches,
            "neon": Neon,
        }
        key = name.lower()
        if key not in registry:
            raise ValueError(
                f"Неизвестный фильтр: {name}. "
                f"Доступные: {', '.join(registry.keys())}"
            )
        return registry[key](**kwargs)

    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет фильтр к изображению и возвращает результат."""
        ...

class Resize(ImageFilter):
    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("Ширина и высота должны быть положительными.")
        self.width = int(width)
        self.height = int(height)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        dst_h, dst_w = self.height, self.width

        x_idx = (np.arange(dst_w) * src_w / dst_w).astype(np.int32)
        y_idx = (np.arange(dst_h) * src_h / dst_h).astype(np.int32)

        x_idx = np.clip(x_idx, 0, src_w - 1)
        y_idx = np.clip(y_idx, 0, src_h - 1)

        return image[y_idx[:, None], x_idx[None, :]]

class RGB2GrayScale(ImageFilter):
    WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        gray = (image.astype(np.float32) * self.WEIGHTS).sum(axis=-1)

        return np.clip(gray, 0, 255).astype(np.uint8)

class Antique(ImageFilter):
    SEPIA = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ], dtype=np.float32)

    def __init__(self, vignette: float = 0.3):
        if not 0.0 <= vignette <= 1.0:
            raise ValueError("vignette должен быть в диапазоне [0, 1].")
        self.vignette = float(vignette)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        sepia = (img[..., None, :] * self.SEPIA[None, None, :, :]).sum(axis=-1)

        if self.vignette > 0:
            sepia *= self._vignette_mask(img.shape[:2])

        return np.clip(sepia, 0, 255).astype(np.uint8)

    def _vignette_mask(self, shape):
        h, w = shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ys = np.arange(h) - cy
        xs = np.arange(w) - cx
        dist2 = ys[:, None] ** 2 + xs[None, :] ** 2
        r_max2 = (ys[0] ** 2 + xs[0] ** 2)
        mask = 1.0 - self.vignette * (dist2 / r_max2)

        return mask[..., None]

class FadeColor(ImageFilter):
    GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)

    def __init__(self, alpha: float = 0.5, contrast: float = 0.85, haze: float = 0.15):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha должен быть в диапазоне [0, 1].")
        if not 0.0 <= contrast <= 1.0:
            raise ValueError("contrast должен быть в диапазоне [0, 1].")
        if not 0.0 <= haze <= 1.0:
            raise ValueError("haze должен быть в диапазоне [0, 1].")
        self.alpha = float(alpha)
        self.contrast = float(contrast)
        self.haze = float(haze)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        gray = (img * self.GRAY_WEIGHTS).sum(axis=-1)

        gray3 = gray[..., None]
        faded = self.alpha * img + (1.0 - self.alpha) * gray3

        faded = faded * self.contrast + (1.0 - self.contrast) * 255.0 * self.haze

        return np.clip(faded, 0, 255).astype(np.uint8)

class Film(ImageFilter):
    IR_MATRIX = np.array([
        [ 0.6, -0.3,  0.1],
        [ 0.2,  0.5,  0.3],
        [ 1.1,  0.3, -0.2],
    ], dtype=np.float32)

    TINT_BGR = np.array([0.95, 1.0, 1.05], dtype=np.float32)

    def __init__(self, grain: float = 8.0, seed: int = None):
        if grain < 0:
            raise ValueError("grain должен быть неотрицательным.")
        self.grain = float(grain)
        self.seed = seed

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        ir = (img[..., None, :] * self.IR_MATRIX[None, None, :, :]).sum(axis=-1)
        ir = ir * self.TINT_BGR[None, None, :]
        if self.grain > 0:
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(loc=0.0, scale=self.grain, size=ir.shape).astype(np.float32)
            ir = ir + noise
        return np.clip(ir, 0, 255).astype(np.uint8)

class Matte(ImageFilter):
    def __init__(self, a_ratio: float = 0.45, b_ratio: float = 0.45, feather: float = 0.05):
        if not 0 < a_ratio <= 0.5 or not 0 < b_ratio <= 0.5:
            raise ValueError("Полуоси должны быть в диапазоне (0, 0.5].")
        if feather < 0:
            raise ValueError("feather должен быть неотрицательным.")
        self.a_ratio = float(a_ratio)
        self.b_ratio = float(b_ratio)
        self.feather = float(feather)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        a = self.a_ratio * w
        b = self.b_ratio * h
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        xs = (np.arange(w) - cx) / a
        ys = (np.arange(h) - cy) / b
        d = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)

        if self.feather > 0:
            mask = np.clip((1.0 + self.feather - d) / (2.0 * self.feather), 0.0, 1.0)
        else:
            mask = (d <= 1.0).astype(np.float32)
        mask = mask[..., None]

        img = image.astype(np.float32)
        result = mask * img + (1.0 - mask) * 255.0
        return np.clip(result, 0, 255).astype(np.uint8)

class Scratches(ImageFilter):
    """Добавляет царапины и шум — эффект состаренной фотографии"""

    def __init__(self,
                 n_scratches: int = 5,
                 scratch_intensity: float = 40.0,
                 noise_level: float = 12.0,
                 seed: int = None):
        if n_scratches < 0:
            raise ValueError("n_scratches должен быть >= 0.")
        if scratch_intensity < 0:
            raise ValueError("scratch_intensity должен быть >= 0.")
        if noise_level < 0:
            raise ValueError("noise_level должен быть >= 0.")
        self.n_scratches = int(n_scratches)
        self.scratch_intensity = float(scratch_intensity)
        self.noise_level = float(noise_level)
        self.seed = seed

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        result = image.astype(np.float32).copy()
        h, w = result.shape[:2]

        # Царапины
        for _ in range(self.n_scratches):
            self._draw_scratches(result, w, h, rng)

        # Шум
        if self.noise_level > 0:
            noise = rng.normal(0.0, self.noise_level, size=result.shape).astype(np.float32)
            result += noise

        return np.clip(result, 0, 255).astype(np.uint8)

    def _draw_scratches(self, image, w, h, rng):
        """Рисует одну случайную царапину (вертикальную, с небольшим наклоном)"""
        x_top = rng.integers(0, w)
        x_bottom = int(x_top + rng.integers(-20, 21))
        x_bottom = int(np.clip(x_bottom, 0, w - 1))

        y_top, y_bottom = 0, h - 1

        # Длина — максимум по вертикали
        length = int(np.hypot(x_bottom - x_top, y_bottom - y_top))
        if length == 0:
            return

        # Параметрические координаты
        ts = np.linspace(0.0, 1.0, length * 2)
        xs = (x_top + ts * (x_bottom - x_top)).astype(np.int32)
        ys = (y_top + ts * (y_bottom - y_top)).astype(np.int32)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        # Светлая или тёмная царапина
        sign = 1.0 if rng.random() < 0.7 else -1.0
        delta = sign * self.scratch_intensity

        image[ys, xs] += delta
        # Утолщаем царапину на 1 пиксель по X, чтобы была видна
        xs2 = np.clip(xs + 1, 0, w - 1)
        image[ys, xs2] += delta * 0.5

class Neon(ImageFilter):
    """Неоновое выделение контуров"""

    # Ядра Собеля
    SOBEL_X = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    SOBEL_Y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)

    def __init__(self,
                 darkness: float = 0.3,
                 neon_color: tuple = (255, 255, 0),   # BGR: циан
                 intensity: float = 1.0):
        if not 0.0 <= darkness <= 1.0:
            raise ValueError("darkness должен быть в [0, 1].")
        if intensity < 0:
            raise ValueError("intensity должен быть >= 0.")
        if len(neon_color) != 3:
            raise ValueError("neon_color должен быть кортежем из 3 чисел (B, G, R).")
        self.darkness = float(darkness)
        self.neon_color = np.array(neon_color, dtype=np.float32)
        self.intensity = float(intensity)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        gray = (img * self.GRAY_WEIGHTS).sum(axis=-1)

        # Свёртка ядрами Собеля
        gx = self._convolve3x3(gray, self.SOBEL_X)
        gy = self._convolve3x3(gray, self.SOBEL_Y)

        # Величина градиента
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Нормализация
        m_max = magnitude.max()
        if m_max > 0:
            magnitude = magnitude / m_max * 255.0

        # Притемнённый оригинал + цветной неон
        darkened = img * self.darkness
        neon_layer = magnitude[..., None] * (self.neon_color / 255.0) * self.intensity
        result = darkened + neon_layer

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _convolve3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Свёртка 2D-изображения с ядром 3x3
        """
        h, w = image.shape
        padded = np.pad(image, pad_width=1, mode='constant', constant_values=0)

        out = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                window = padded[i:i + h, j:j + w]
                out += kernel[i, j] * window
        return out