import numpy as np
import cv2
from abc import ABC
from numpy.lib.stride_tricks import sliding_window_view


class ImageFilter(ABC):
    _registry: dict[str, type["ImageFilter"]] = {}
    _cli_args = None

    @classmethod
    def bind_args(cls, args) -> None:
        cls._cli_args = args

    def __init_subclass__(cls, filter_name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if filter_name is not None:
            ImageFilter._registry[filter_name] = cls

    @staticmethod
    def get_filter(name: str, *args, **kwargs) -> "ImageFilter":
        try:
            cls = ImageFilter._registry[name]
        except KeyError:
            raise ValueError(
                f"Unknown filter: {name}. "
                f"Available: {', '.join(ImageFilter._registry)}"
            )
            
        return cls(*args, **kwargs)

    def apply_filter(self, image):
        pass


class ResizeFilter(ImageFilter, filter_name='resize'):
    def __init__(self):
        self.width = self._cli_args.width
        self.height = self._cli_args.height

    def apply_filter(self, image):
        if self.width is None or self.height is None:
            raise ValueError("Resize filter requires width and height")
        h, w = image.shape[:2]
        new_h, new_w = self.height, self.width

        row_indices = (np.arange(new_h) * (h / new_h)).astype(int)
        col_indices = (np.arange(new_w) * (w / new_w)).astype(int)

        resized = image[row_indices[:, None], col_indices[None, :]]
        return resized


class GrayscaleFilter(ImageFilter, filter_name='grayscale'):
    def apply_filter(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray


class AntiqueFilter(ImageFilter, filter_name='antique'):
    def apply_filter(self, image):
        sepia_matrix = np.array(
            [
                [0.131, 0.534, 0.272],
                [0.168, 0.686, 0.349],
                [0.189, 0.769, 0.393],
            ]
        )
        img = image.astype(np.float32) / 255.0
        sepia = img.astype(np.float32).dot(sepia_matrix.T)

        h, w = sepia.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        vignette = 1 - 0.9 * (dist / max_dist) ** 2
        vignette = vignette[:, :, np.newaxis]
        sepia = sepia * vignette

        noise = np.random.normal(0, 0.02, sepia.shape)
        sepia = np.clip(sepia + noise, 0, 1)

        return (sepia * 255).astype(np.uint8)


class FadeFilter(ImageFilter, filter_name='fade'):
    def __init__(self):
        self.contrast = self._cli_args.contrast
        self.alpha = self._cli_args.alpha

    def apply_filter(self, image):
        img = image.astype(np.float32) / 255.0

        brightness = 0.2
        faded = (img - 0.5) * self.contrast + 0.5 + brightness

        white = np.ones_like(faded)
        faded = self.alpha * faded + (1 - self.alpha) * white
        faded = np.clip(faded, 0, 1)

        return (faded * 255).astype(np.uint8)


class FilmFilter(ImageFilter, filter_name='film'):
    def __init__(self):
        self.grain = self._cli_args.grain

    def apply_filter(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        film = np.stack([b, r, g], axis=2)

        grain = np.random.normal(0, self.grain, film.shape)
        film = np.clip(film + grain, 0, 255)

        return (film).astype(np.uint8)


class MatteFilter(ImageFilter, filter_name='matte'):
    def __init__(self):
        self.radius = self._cli_args.radius
        self.softness = self._cli_args.softness

    def apply_filter(self, image):
        img = image.astype(np.float32) / 255.0
        h, w = img.shape[:2]

        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        norm = min(h, w) / 2
        d = dist / norm

        t = np.clip((d - self.radius) / self.softness, 0, 1)
        t = t * t * (3 - 2 * t)
        t = t[:, :, np.newaxis]

        img = img * (1 - t) + t * 1.0

        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)
        return (img * 255).astype(np.uint8)


class ScratchesFilter(ImageFilter, filter_name='scratches'):
    def __init__(self):
        self.num_scratches = self._cli_args.num_scratches
        self.noise_amount = self._cli_args.noise_amount

    def apply_filter(self, image):
        old_im = AntiqueFilter().apply_filter(image)        
        img = old_im.copy()
        h, w = img.shape[:2]

        for _ in range(self.num_scratches):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

        noise = np.random.rand(h, w)
        salt = noise > 1 - self.noise_amount / 2
        pepper = noise < self.noise_amount / 2
        img[salt] = [255, 255, 255]
        img[pepper] = [0, 0, 0]

        return img


class Neon(ImageFilter, filter_name='neon'):
    def __init__(self, color=(255, 255, 50)):
        self.threshold = self._cli_args.threshold
        self.glow = self._cli_args.glow
        self.color = color
        self.glow_width = self._cli_args.glow_width
        self.line_width = self._cli_args.line_width

    @staticmethod
    def _dilate(mask):
        out = mask.copy()
        out[1:, :] = np.maximum(out[1:, :], mask[:-1, :])
        out[:-1, :] = np.maximum(out[:-1, :], mask[1:, :])
        out[:, 1:] = np.maximum(out[:, 1:], mask[:, :-1])
        out[:, :-1] = np.maximum(out[:, :-1], mask[:, 1:])
        return out

    def _glow_by_dilation(self, mask, sigma):
        glow = mask * 0.4
        current = mask
        radius = max(1, int(sigma * 3))
        for r in range(1, radius + 1):
            current = self._dilate(current)
            weight = np.exp(-(r * r) / (2 * sigma * sigma)) * 0.4
            glow = np.maximum(glow, current * weight)
        return glow

    def apply_filter(self, image):
        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2126 * r + 0.7152 * g + 0.0722 * b

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)

        mask = (mag > self.threshold).astype(np.float32)

        for _ in range(max(0, self.line_width - 1)):
            mask = self._dilate(mask)

        sigma = max(1.0, float(self.glow_width))
        glow_mask = self._glow_by_dilation(mask, sigma)

        cb, cg, cr = self.color
        base = img * 0.65

        edges = np.zeros_like(img)
        edges[:, :, 0] = mask * cb
        edges[:, :, 1] = mask * cg
        edges[:, :, 2] = mask * cr

        glow_layer = np.zeros_like(img)
        glow_layer[:, :, 0] = glow_mask * cb
        glow_layer[:, :, 1] = glow_mask * cg
        glow_layer[:, :, 2] = glow_mask * cr

        out = base + edges + glow_layer * self.glow
        return np.clip(out, 0, 255).astype(np.uint8)