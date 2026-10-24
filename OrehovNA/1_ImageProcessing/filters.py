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
        return np.clip(antique, 0, 255).astype(np.uint8)

class FadeColorFilter(ImageFilter):
    def __init__(self, factor=0.4):
        self.factor = float(factor)

    def apply_filter(self, image):
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        gray_3d = np.stack([gray, gray, gray], axis=-1)
        faded = image * (1 - self.factor) + gray_3d * self.factor
        return np.clip(faded, 0, 255).astype(np.uint8)

class FilmSimulationFilter(ImageFilter):
    def apply_filter(self, image):
        r = image[:, :, 2].astype(float)
        g = image[:, :, 1].astype(float)
        
        ir_gray = r * 0.7 + g * 0.3
        
        ir_gray = (ir_gray - 128) * 1.5 + 128
        ir_gray = np.clip(ir_gray, 0, 255).astype(np.uint8)
        return np.stack([ir_gray, ir_gray, ir_gray], axis=-1)

class MatteFilter(ImageFilter):
    def __init__(self, matte_radius=0.7, matte_softness=0.2):
        self.radius = float(matte_radius)
        self.softness = float(matte_softness)
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

        res = image.astype(np.float32) * (1 - weight) + 255.0 * weight
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
    def __init__(self, neon_strength: float = 1.5, neon_blur: float = 0.8):
        self.strength = float(neon_strength)
        self.blur_factor = float(neon_blur)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w, c = image.shape
        gray = GrayScaleFilter().apply_filter(image)[:, :, 0].astype(np.float32)

        gx, gy = np.zeros_like(gray), np.zeros_like(gray)
        gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        gy[:-1, :] = gray[1:, :] - gray[:-1, :]
        
        edges = np.clip(np.sqrt(gx**2 + gy**2) / 255.0, 0.0, 1.0)

        def multi_blur(img_mask, iterations):
            res = img_mask.copy()
            for _ in range(max(0, iterations)):
                p = np.pad(res, 1, mode="edge")
                res = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                       p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                       p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9.0
            return res

        narrow_glow = multi_blur(edges, round(3 * self.blur_factor))
        wide_glow = multi_blur(edges, round(8 * self.blur_factor))

        glow_map = self.strength * (edges * 60.0 + narrow_glow * 120.0 + wide_glow * 170.0)

        res = image.astype(np.float32)
        
        res[:, :, 0] += glow_map
        res[:, :, 1] += glow_map

        return np.clip(res, 0, 255).astype(np.uint8)



