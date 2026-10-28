import numpy as np
from abc import ABC, abstractmethod


class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def get_filter(filter_type: str, **kwargs) -> 'ImageFilter':
        filters = {
            "resize": Resize,
            "grayscale": Grayscale,
            "antique": Antique,
            "fade": FadeColor,
            "infrared": Infrared,
            "matte": Matte,
            "aged": AgedPhoto,
            "neon": Neon
        }

        filter_name = filter_type.lower()
        if filter_name not in filters:
            raise ValueError(f"Неизвестный тип фильтра: '{filter_type}'. "
                             f"Доступные: {list(filters.keys())}")

        return filters[filter_name](**kwargs)



class Resize(ImageFilter):

    def __init__(self, width: int = None, height: int = None, scale: float = None):
        self.width = width
        self.height = height
        self.scale = scale

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h_src, w_src = image.shape[:2]

        if self.scale is not None:
            w_new = max(1, int(w_src * self.scale))
            h_new = max(1, int(h_src * self.scale))
        else:
            w_new = self.width if self.width else w_src
            h_new = self.height if self.height else h_src

        x_idx = (np.arange(w_new) * w_src / w_new).astype(np.int32)
        y_idx = (np.arange(h_new) * h_src / h_new).astype(np.int32)
        x_idx = np.clip(x_idx, 0, w_src - 1)
        y_idx = np.clip(y_idx, 0, h_src - 1)

        return image[y_idx][:, x_idx]



class Grayscale(ImageFilter):
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        b, g, r = image[:, :, 0].astype(np.float32), image[:, :, 1].astype(np.float32), image[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)


class Antique(ImageFilter):
    def __init__(self, vignette_strength: float = 0.5):
        self.vignette_strength = max(0.0, min(1.0, float(vignette_strength)))

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        gray = Grayscale().apply_filter(image).astype(np.float32)
        sepia = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

        sepia[:, :, 0] = np.clip(gray * 0.43, 0, 255)   # B
        sepia[:, :, 1] = np.clip(gray * 0.74, 0, 255)   # G
        sepia[:, :, 2] = np.clip(gray * 1.07, 0, 255)   # R

        if self.vignette_strength > 0:
            h, w = sepia.shape[:2]
            cy, cx = h / 2.0, w / 2.0
            rx, ry = cx, cy

            y_idx, x_idx = np.ogrid[:h, :w]
            d = np.sqrt(((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2)

            mask = np.clip((d - 0.4) / 0.6, 0.0, 1.0)
            mask = (mask ** 1.5) * self.vignette_strength
            mask = mask[..., np.newaxis]

            sepia = sepia.astype(np.float32) * (1.0 - mask)
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        return sepia

class FadeColor(ImageFilter):
    def __init__(self, alpha: float = 0.5):
        self.alpha = max(0.0, min(1.0, float(alpha)))
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32) * (1.0 - self.alpha) + 255.0 * self.alpha
        return np.clip(result, 0, 255).astype(np.uint8)


class Infrared(ImageFilter):

    def __init__(self, noise_level: int = 10):
        self.noise_level = noise_level

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        new_r = b
        new_g = np.clip(g * 1.2, 0, 255)
        new_b = r

        result = np.stack([new_b, new_g, new_r], axis=2)

        noise = np.random.normal(0, self.noise_level, result.shape)
        result = np.clip(result + noise, 0, 255)
        return result.astype(np.uint8)


class Matte(ImageFilter):
    def __init__(self, radius: float = 0.85, softness: float = 0.15):
        self.radius = max(0.1, min(1.0, float(radius)))
        self.softness = max(0.001, min(0.5, float(softness)))

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        rx, ry = cx * self.radius, cy * self.radius

        y_idx, x_idx = np.ogrid[:h, :w]
        d = np.sqrt(((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2)

        mask = np.clip((d - (1.0 - self.softness)) / (2 * self.softness), 0, 1)
        mask = mask[..., np.newaxis]

        result = image.astype(np.float32) * (1 - mask) + 255.0 * mask
        return np.clip(result, 0, 255).astype(np.uint8)


class AgedPhoto(ImageFilter):

    def apply_filter(self, img, **kwargs):
        noise = kwargs.get("noise", 0.1)
        scratches = kwargs.get("scratches", 10)


        res = img.astype(np.float32)
        h, w = res.shape[:2]

        res += np.random.normal(0, noise * 255, (h, w, 1))

        for _ in range(scratches):
            x_line = np.random.randint(0, w)
            res[:, x_line] = res[:, x_line] * 0.85 + 255 * 0.15

        return np.clip(res, 0, 255).astype(np.uint8)
        
class Neon(ImageFilter):
    def __init__(self, threshold: float = 0.04, glow: float = 1.0,
                 dim: float = 0.55, color: tuple = (255, 255, 50),
                 glow_width: float = 4.0):
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.glow = float(glow)
        self.dim = max(0.1, min(1.0, float(dim)))
        self.color = tuple(color)
        self.glow_width = max(1.0, float(glow_width))

    def _sobel(self, gray: np.ndarray) -> np.ndarray:
        p = np.pad(gray, 1, mode='edge')
        gx = (-1 * p[:-2, :-2] + 1 * p[:-2, 2:] +
              -2 * p[1:-1, :-2] + 2 * p[1:-1, 2:] +
              -1 * p[2:, :-2] + 1 * p[2:, 2:])
        gy = (-1 * p[:-2, :-2] - 2 * p[:-2, 1:-1] - 1 * p[:-2, 2:] +
               1 * p[2:, :-2] + 2 * p[2:, 1:-1] + 1 * p[2:, 2:])
        return np.sqrt(gx**2 + gy**2)

    @staticmethod
    def _dilate(mask: np.ndarray) -> np.ndarray:
        p = np.pad(mask, 1, mode='constant', constant_values=0)
        h, w = mask.shape
        out = np.zeros_like(mask)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                out = np.maximum(out, p[1+dy:1+dy+h, 1+dx:1+dx+w])
        return out

    def _glow_by_dilation(self, mask: np.ndarray, sigma: float) -> np.ndarray:
        glow = mask * 0.5
        current = mask
        radius = max(1, int(sigma * 1.5))
        for r in range(1, radius + 1):
            current = self._dilate(current)
            weight = np.exp(-(r * r) / (2 * (sigma / 2) ** 2)) * 0.9
            glow = np.maximum(glow, current * weight)
        return glow

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        gray = Grayscale().apply_filter(image).astype(np.float32)

        edges = self._sobel(gray)
        max_val = edges.max()
        if max_val > 0:
            edges = edges / max_val

        mask = (edges > self.threshold).astype(np.float32)
        glow_mask = self._glow_by_dilation(mask, self.glow_width)

        if len(image.shape) == 2:
            base = np.stack((image,) * 3, axis=-1).astype(np.float32)
        else:
            base = image[:, :, :3].astype(np.float32)
        base = base * self.dim

        cb, cg, cr = self.color
        edges_layer = np.zeros_like(base)
        edges_layer[:, :, 0] = mask * cb
        edges_layer[:, :, 1] = mask * cg
        edges_layer[:, :, 2] = mask * cr

        glow_layer = np.zeros_like(base)
        glow_layer[:, :, 0] = glow_mask * cb
        glow_layer[:, :, 1] = glow_mask * cg
        glow_layer[:, :, 2] = glow_mask * cr

        result = base + edges_layer + glow_layer * self.glow
        result = np.clip(result, 0, 255).astype(np.uint8)

        if len(image.shape) == 3 and image.shape[2] == 4:
            result = np.dstack((result, image[:, :, 3]))

        return result