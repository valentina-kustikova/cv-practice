import cv2
import numpy as np

class ImageFilter:
    @staticmethod
    def get_filter(filter_type, **kwargs):
        filters = {
            'resize': ResizeFilter,
            'gray': GrayScaleFilter,
            'antique': AntiqueFilter,
            'fade': FadeColorFilter,
            'film': FilmSimulationFilter,
            'matte': MatteFilter,
            'scratch': ScratchNoiseFilter,
            'neon': NeonFilter
        }
        if filter_type not in filters:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return filters[filter_type](**kwargs)

    def apply_filter(self, image):
        raise NotImplementedError("Each filter must implement apply_filter method.")

class ResizeFilter(ImageFilter):
    def __init__(self, scale=None, width=None, height=None):
        self.scale = float(scale) if scale else None
        self.width = int(width) if width else None
        self.height = int(height) if height else None

    def apply_filter(self, image):
        h, w, c = image.shape
        if self.scale:
            new_h, new_w = int(h * self.scale), int(w * self.scale)
        elif self.width and self.height:
            new_h, new_w = self.height, self.width
        else:
            return image

        x_ratio = float(w) / new_w
        y_ratio = float(h) / new_h

        y_indices = np.arange(new_h)
        x_indices = np.arange(new_w)
        
        y_l = (y_ratio * y_indices).astype(int)
        x_l = (x_ratio * x_indices).astype(int)
        
        y_h = np.minimum(y_l + 1, h - 1)
        x_h = np.minimum(x_l + 1, w - 1)

        y_weight = ((y_ratio * y_indices) - y_l)[:, None, None]
        x_weight = ((x_ratio * x_indices) - x_l)[None, :, None]

        a = image[y_l[:, None], x_l, :]
        b = image[y_l[:, None], x_h, :]
        c_val = image[y_h[:, None], x_l, :]
        d = image[y_h[:, None], x_h, :]
        
        pixel_val = (a * (1 - x_weight) * (1 - y_weight) +
                     b * x_weight * (1 - y_weight) +
                     c_val * (1 - x_weight) * y_weight +
                     d * x_weight * y_weight)
                     
        return np.clip(pixel_val, 0, 255).astype(np.uint8)


class GrayScaleFilter(ImageFilter):
    def apply_filter(self, image):
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

class AntiqueFilter(ImageFilter):
    def apply_filter(self, image):
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        
        output_r = r * 0.393 + g * 0.769 + b * 0.189
        output_g = r * 0.349 + g * 0.686 + b * 0.168
        output_b = r * 0.272 + g * 0.534 + b * 0.131
        
        antique = np.stack([output_b, output_g, output_r], axis=-1)
        antique = np.clip(antique, 0, 255).astype(np.uint8)
        return MatteFilter(1.2, 0.9, 0.0).apply_filter(antique)

class FadeColorFilter(ImageFilter):
    def __init__(self, factor=0.4):
        self.factor = float(factor)

    def apply_filter(self, image):
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        gray_3d = np.stack([gray, gray, gray], axis=-1)
        faded = image * (1 - self.factor) + gray_3d * self.factor
        return np.clip(faded, 0, 255).astype(np.uint8)

import numpy as np

class FilmSimulationFilter(ImageFilter):
    def apply_filter(self, image):
        b = image[:, :, 0].astype(float)
        g = image[:, :, 1].astype(float)
        r = image[:, :, 2].astype(float)
        
        ir_light = r * 0.3 + g * 0.7
        
        # 3. Поднимаем контраст для эффекта «серебряного» свечения листьев
        ir_light = (ir_light - 128) * 1.5 + 128
        ir_light = np.clip(ir_light, 0, 255).astype(np.uint8)
        
        # 4. Добавляем мягкую подложку для теней (аналоговая вуаль), чтобы они не были черными
        shadow_r = np.full_like(ir_light, 20)
        
        # 5. Собираем каналы в правильном BGR порядке, чтобы на выходе получить красный цвет:
        # Чтобы картинка была красной, Синий и Зеленый каналы должны быть темными (зануляем их)
        blue_channel = np.zeros_like(ir_light)
        green_channel = np.zeros_like(ir_light)
        
        # Направляем основной ИК-сигнал строго в Красный канал
        red_channel = np.clip(ir_light.astype(int) + shadow_r, 0, 255).astype(np.uint8)
        
        # Собираем финальный массив в исходном формате BGR
        return np.stack([blue_channel, green_channel, red_channel], axis=-1)


class MatteFilter(ImageFilter):
    def __init__(self, matte_radius=0.7, matte_softness=0.2, color = 255.0):
        self.radius = float(matte_radius)
        self.softness = float(matte_softness)
        self.color = float(color)
        if self.radius <= 0:
            raise ValueError("matte_radius must be positive")
        if self.softness < 0:
            raise ValueError("matte_softness must be non-negative")

    def apply_filter(self, image):
        h, w = image.shape[:2]
        y_c, x_c = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]

        d = (x - x_c) ** 2 / (x_c ** 2) + (y - y_c) ** 2 / (y_c ** 2)

        if self.softness == 0:
            weight = (d > self.radius).astype(np.float32)
        else:
            weight = np.clip((d - self.radius) / self.softness, 0, 1).astype(np.float32)
        
        weight = weight[:, :, None]

        res = image.astype(np.float32) * (1 - weight) + self.color * weight
        return np.clip(res, 0, 255).astype(np.uint8)


class ScratchNoiseFilter(ImageFilter):
    def __init__(self, noise: float = 0.04, scratches: int = 7):
        self.noise = float(noise)
        self.scratches = int(scratches)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w, c = image.shape
        res = image.astype(np.float32) + np.random.normal(0.0, self.noise * 255.0, (h, w, 1))
        
        mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(self.scratches):
            x0, y0 = np.random.randint(0, w), np.random.randint(0, int(h * 0.5))
            length = np.random.randint(int(h * 0.1), int(h * 0.4))
            angle = np.random.uniform(-0.08, 0.08)
            
            t = np.linspace(0, 1, length)
            xs = np.clip(x0 + t * length * np.sin(angle) + np.random.normal(0, 0.4, length), 0, w - 1).astype(np.int32)
            ys = np.clip(y0 + t * length * np.cos(angle) + np.random.normal(0, 0.4, length), 0, h - 1).astype(np.int32)
            
            mask[ys, xs] = np.maximum(mask[ys, xs], np.random.uniform(0.3, 0.6))

        p = np.pad(mask, 1, mode="edge")
        mask = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0

        res = res * (1.0 - mask[..., None]) + np.random.uniform(210, 240) * mask[..., None]
        res = np.clip(res, 0, 255)

        res_uint8 = np.clip(res, 0, 255).astype(np.uint8)
        
        sepia = AntiqueFilter().apply_filter(res_uint8)
        
        return np.clip(res * 0.4 + sepia * 0.6, 0, 255).astype(np.uint8)


class NeonFilter(ImageFilter):
    def __init__(
        self,
        neon_strength: float = 1.5,
        neon_blur: float = 0.8,
        neon_color=(255, 0, 255),
        tint=(120, 0, 80),
        glow_inner: float = 0.1,        # было 0.9
        glow_outer: float = 1.5,         # было 0.7
        glow_inner_scale: float = 3.0,   # было 2.0 — сильнее размытие ближнего ореола
        glow_outer_scale: float = 6.0,   # было 5.0
    ):
        self.strength = float(neon_strength)
        self.blur_factor = float(neon_blur)
        self.color = self._norm_color(neon_color)
        self.tint = self._norm_color(tint)

        self.glow_inner = float(glow_inner)
        self.glow_outer = float(glow_outer)
        self.glow_inner_scale = float(glow_inner_scale)
        self.glow_outer_scale = float(glow_outer_scale)

    @staticmethod
    def _norm_color(c):
        c = np.asarray(c, dtype=np.float32).reshape(3)
        if c.max() > 1.0:
            c = c / 255.0
        return c

    @staticmethod
    def _gaussian_kernel(sigma):
        sigma = max(float(sigma), 1e-6)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        return k / k.sum()

    @staticmethod
    def _conv_axis(img, kernel, axis):
        n = img.shape[axis]
        r = len(kernel) // 2
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(img, pad, mode="edge")

        out = np.zeros_like(img, dtype=np.float32)
        for i, w in enumerate(kernel):
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(i, i + n)
            out += float(w) * p[tuple(sl)]
        return out

    @classmethod
    def _blur(cls, img, sigma):
        if sigma <= 0:
            return img.astype(np.float32, copy=False)
        k = cls._gaussian_kernel(sigma)
        out = cls._conv_axis(img, k, 0)
        out = cls._conv_axis(out, k, 1)
        return out

    @staticmethod
    def _sobel(gray):
        p = np.pad(gray.astype(np.float32), 1, mode="edge")

        gx = (
            p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
            - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]
        )
        gy = (
            p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
            - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
        )
        return gx, gy

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        img = image.astype(np.float32)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if img.max() > 1.0:
            img /= 255.0

        # Яркость
        lum = (
            0.114 * img[:, :, 0]
            + 0.587 * img[:, :, 1]
            + 0.299 * img[:, :, 2]
        )

        # Слегка размываем перед выделением контуров, чтобы не ловить шум
        smooth = self._blur(lum, max(0.6, self.blur_factor))

        # Контуры
        gx, gy = self._sobel(smooth)
        mag = np.sqrt(gx * gx + gy * gy)

        max_v = float(mag.max())
        if max_v > 1e-8:
            mag /= max_v

        # “Ядро” неона — тонкие контуры
        edge = np.clip((mag - 0.03) / 0.25, 0.0, 1.0)
        edge = edge ** 0.65
        core = np.clip(edge * self.strength, 0.0, 1.0)

        # Свечение вокруг контуров: несколько радиусов размытия
        glow_1 = self._blur(core, max(0.8, self.blur_factor * self.glow_inner_scale))
        glow_2 = self._blur(core, max(2.0, self.blur_factor * self.glow_outer_scale))
        glow = np.clip((self.glow_inner * glow_1 + self.glow_outer * glow_2) * self.strength,0.0, 1.0)

        # Широкая аура
        aura = self._blur(core, max(3.0, self.blur_factor * 10.0))
        aura = np.clip(aura * self.strength * 0.8, 0.0, 1.0)

        # Затемнённая и подкрашенная база
        tint = self.tint.reshape(1, 1, 3)
        ambient = (0.12 + 0.35 * lum[:, :, None]) * tint
        base = np.clip(img * 0.22 + ambient, 0.0, 1.0)

        # Цветной неоновый слой
        color = self.color.reshape(1, 1, 3)
        neon_layer = color * (
            core[:, :, None] * 1.8
            + glow[:, :, None] * 1.15
            + aura[:, :, None] * 0.75
        )
        neon_layer = np.clip(neon_layer, 0.0, 1.0)

        # Screen-смешивание
        out = 1.0 - (1.0 - base) * (1.0 - neon_layer)
        out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)

        return out




