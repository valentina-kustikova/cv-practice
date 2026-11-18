from abc import ABC, abstractmethod
import numpy as np


# Приводит серый/1-канальный/BGRA к BGR
def _to_bgr(image):
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


# Свёртка 3x3, сдвигаем изображение и умножаем на элементы ядра
def _convolve2d(channel, kernel):
    h, w = channel.shape
    padded = np.pad(channel, 1, mode="edge")
    out = np.zeros((h, w), dtype=np.float32)

    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            out += padded[i:i + h, j:j + w] * kernel[i, j]

    return out


# Виньетка
def _radial_factor(h: int, w: int, vignette: float, center_gain: float = 0.0):
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.ogrid[:h, :w]

    nx = (x - cx) / max(cx, 1e-6)
    ny = (y - cy) / max(cy, 1e-6)
    r = np.clip(np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0), 0.0, 1.0)

    return 1.0 + center_gain * (1.0 - r) - vignette * (r ** 2)


# Абстрактный фильтр
class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, image):
        pass

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> "ImageFilter":
        filter_type = filter_type.lower()

        if filter_type in ("grayscale", "gray"):
            return BGR2GrayScale()
        elif filter_type in ("antique", "sepia"):
            return AntiqueFilter()
        elif filter_type == "resize":
            width = kwargs.get("width")
            height = kwargs.get("height")
            if width is None or height is None or width <= 0 or height <= 0:
                raise ValueError("Для resize требуются положительные --width и --height")
            return ResizeFilter(int(width), int(height))
        elif filter_type == "fade":
            alpha = float(kwargs.get("alpha", 0.6))
            return FadeFilter(alpha)
        elif filter_type in ("film", "infrared"):
            return FilmFilter()
        elif filter_type == "matte":
            radius = float(kwargs.get("radius", 0.85))
            return MatteFilter(radius=radius)
        elif filter_type == "scratches":
            noise_ratio = float(kwargs.get("noise_ratio", 0.01))
            num_lines = int(kwargs.get("num_lines", 6))
            return ScratchesNoiseFilter(noise_ratio, num_lines)
        elif filter_type == "neon":
            return NeonFilter()
        else:
            raise ValueError(f"Неизвестный тип фильтра: '{filter_type}'")


# resize x = x' * W / W', y = y' * H / H'
class ResizeFilter(ImageFilter):
    def __init__(self, target_width: int, target_height: int):
        self.target_width = target_width
        self.target_height = target_height

    def apply_filter(self, image):
        orig_height, orig_width = image.shape[:2]

        row_indices = (np.arange(self.target_height) * (orig_height / self.target_height)).astype(int)
        col_indices = (np.arange(self.target_width) * (orig_width / self.target_width)).astype(int)

        row_indices = np.clip(row_indices, 0, orig_height - 1)
        col_indices = np.clip(col_indices, 0, orig_width - 1)

        return image[np.ix_(row_indices, col_indices)]


# Оттенки серого: Y = 0.299*R + 0.587*G + 0.114*B
class BGR2GrayScale(ImageFilter):
    def apply_filter(self, image):
        if image.ndim == 2:
            return image.copy()

        img = _to_bgr(image)
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return np.clip(gray, 0, 255).astype(np.uint8)


# Антиквариат: сепия + виньетка
class AntiqueFilter(ImageFilter):
    def __init__(self, vignette: float = 0.75, center_gain: float = 0.5):
        self.vignette = float(np.clip(vignette, 0.0, 0.9))
        self.center_gain = float(np.clip(center_gain, 0.0, 0.5))

    def apply_filter(self, image):
        img = _to_bgr(image)
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)

        r_new = 0.393 * r + 0.769 * g + 0.189 * b
        g_new = 0.349 * r + 0.686 * g + 0.168 * b
        b_new = 0.272 * r + 0.534 * g + 0.131 * b
        result = np.stack([b_new, g_new, r_new], axis=-1)

        h, w = result.shape[:2]
        factor = _radial_factor(h, w, self.vignette, self.center_gain)[:, :, None]
        result = result * factor

        return np.clip(result, 0, 255).astype(np.uint8)

# Выцветание I' = a*I + (1-a)*255
class FadeFilter(ImageFilter):
    def __init__(self, alpha: float = 0.6):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def apply_filter(self, image):
        img_float = image.astype(np.float32)
        faded = img_float * self.alpha + (1.0 - self.alpha) * 255.0
        return np.clip(faded, 0, 255).astype(np.uint8)

# ИК-плёнка V = 2G - R - B, IR = 128 + k*V
class FilmFilter(ImageFilter):
    def __init__(self, gain: float = 4.0):
        self.gain = float(gain)

    def apply_filter(self, image):
        img = _to_bgr(image)
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        veg = 2.0 * g - r - b
        ir = np.clip(128.0 + self.gain * veg, 0, 255)

        b_new = np.clip(b * 0.60, 0, 255)
        g_new = np.clip(r * 0.35, 0, 255)
        r_new = ir

        result = np.stack([b_new, g_new, r_new], axis=-1)
        return np.clip(result, 0, 255).astype(np.uint8)


# Matte эффект белой виньетки
class MatteFilter(ImageFilter):
    def __init__(self, radius: float = 0.85, feather: float = 0.3):
        self.radius = float(np.clip(radius, 0.05, 1.0))
        self.feather = float(np.clip(feather, 0.01, 1.0))

    def apply_filter(self, image):
        h, w = image.shape[:2]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

        a = max(cx * self.radius, 1e-6)
        b = max(cy * self.radius, 1e-6)
        y, x = np.ogrid[:h, :w]

        dist = ((x - cx) ** 2) / (a * a) + ((y - cy) ** 2) / (b * b)
        start = 1.0 - self.feather
        mask = np.clip((dist - start) / self.feather, 0.0, 1.0)

        if image.ndim == 3:
            mask = mask[:, :, None]

        result = image.astype(np.float32) * (1.0 - mask) + 255.0 * mask
        return np.clip(result, 0, 255).astype(np.uint8)

class ScratchesNoiseFilter(ImageFilter):
    def __init__(self, noise_ratio: float = 0.01, num_lines: int = 6):
        self.noise_ratio = float(noise_ratio)
        self.num_lines = int(num_lines)

    def apply_filter(self, image):
        res = image.astype(np.float32)
        h, w = res.shape[:2]

        # N = noise_ratio * H * W
        num_noise = int(self.noise_ratio * h * w)
        if num_noise > 0:
            ys = np.random.randint(0, h, num_noise)
            xs = np.random.randint(0, w, num_noise)
            tone = np.random.randint(120, 226, num_noise).astype(np.float32)
            if res.ndim == 3:
                tone = tone[:, None]
            res[ys, xs] = res[ys, xs] * 0.5 + tone * 0.5
        max_len = max(1, int(min(h, w) * 0.6))
        for _ in range(max(0, self.num_lines)):
            x0 = np.random.randint(0, w)
            y0 = np.random.randint(0, h)
            length = np.random.randint(max(8, max_len // 3), max(9, max_len))

            angle = np.random.uniform(0.0, np.pi)
            dx, dy = np.cos(angle), np.sin(angle)

            # Перпендикуляр
            px, py = -dy, dx
            bend = np.random.uniform(-0.12, 0.12) * length

            t = np.linspace(0.0, 1.0, length)
            wobble = bend * np.sin(np.pi * t)
            xs = np.clip((x0 + dx * length * t + px * wobble).astype(np.int32), 0, w - 1)
            ys = np.clip((y0 + dy * length * t + py * wobble).astype(np.int32), 0, h - 1)
            # Прерывистость
            keep = np.random.rand(length) > 0.2
            xs, ys = xs[keep], ys[keep]
            if xs.size == 0:
                continue

            # Полупрозрачное смешивание I' = (1-a)*I + a*tone
            tone = np.float32(np.random.randint(170, 221))
            alpha = np.float32(np.random.uniform(0.25, 0.5))
            res[ys, xs] = res[ys, xs] * (1.0 - alpha) + tone * alpha

        return np.clip(res, 0, 255).astype(np.uint8)


# Неон
class NeonFilter(ImageFilter):
    def __init__(self, gain: float = 3.0):
        self.gain = float(gain)

    def apply_filter(self, image):
        base = _to_bgr(image).astype(np.float32)
        b = base[:, :, 0]
        g = base[:, :, 1]
        r = base[:, :, 2]
        gray = 0.114 * b + 0.587 * g + 0.299 * r

        #ядро собеля
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        gx = _convolve2d(gray, kernel_x)
        gy = _convolve2d(gray, kernel_y)
        # |grad| = sqrt(Gx^2 + Gy^2)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        edges = (np.clip(magnitude / 4.0, 0, 255) / 255.0) ** 0.8

        box = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
        edges_glow = _convolve2d(edges, box)
        glow_map = np.clip(edges + 0.7 * edges_glow, 0.0, 1.0)

        
        neon_bgr = np.array([1.0, 0.8, 0.2], dtype=np.float32)
        glow = glow_map[:, :, None] * neon_bgr * (255.0 * self.gain)

        
        #свечение на исходное изображение
        result = base * 0.15 + glow
        return np.clip(result, 0, 255).astype(np.uint8)
