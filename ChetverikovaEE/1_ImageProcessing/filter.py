import numpy as np
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
        self.alpha = self._cli_args.alpha

    def apply_filter(self, image):
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        a = w * self.alpha
        b = h * self.alpha

        mask = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1

        result = image.copy()
        result[~mask] = [255, 255, 255] 
        return result


class ScratchesFilter(ImageFilter, filter_name='scratches'):
    def __init__(self):
        self.num_scratches = self._cli_args.num_scratches
        self.noise_amount = self._cli_args.noise_amount

    def apply_filter(self, image):
        img = image.copy()
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


class NeonFilter(ImageFilter, filter_name='neon'):
    def __init__(self):
        self.threshold = self._cli_args.threshold
        self.glow_ksize = self._cli_args.glow_ksize

    @staticmethod
    def convolve2d(img, kernel):
                kh, kw = kernel.shape
                padded = np.pad(img, ((kh//2, kh//2), (kw//2, kw//2)), mode='edge')
                windows = sliding_window_view(padded, (kh, kw))
                return np.sum(windows * kernel, axis=(-1, -2))

    @staticmethod
    def blur_channel(ch, kernel_1d, k):
        pad = k // 2
        padded = np.pad(ch, ((0, 0), (pad, pad)), mode='edge')
        blurred = np.zeros_like(ch)
        for i in range(k):
            blurred += kernel_1d[i] * padded[:, i:i+ch.shape[1]]
        padded = np.pad(blurred, ((pad, pad), (0, 0)), mode='edge')
        blurred = np.zeros_like(ch)
        for i in range(k):
            blurred += kernel_1d[i] * padded[i:i+ch.shape[0], :]
        return blurred

    def apply_filter(self, image):
        gray = GrayscaleFilter().apply_filter(image)
    
        gray_f = gray.astype(np.float32)

        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)


        gx = self.convolve2d(gray_f, Kx)
        gy = self.convolve2d(gray_f, Ky)

        magnitude = np.sqrt(gx * gx + gy * gy)
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max() * 255.0
        magnitude = magnitude.astype(np.uint8)

        edges = (magnitude > self.threshold).astype(np.uint8) * 255

        neon = np.zeros_like(image)
        if neon.ndim == 2:
            neon = np.stack([neon] * 3, axis=-1)
        neon[edges > 0] = (255, 255, 0)

        k = self.glow_ksize
        sigma = k / 6.0
        x = np.arange(k) - (k - 1) / 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()


        glow = np.zeros_like(neon, dtype=np.float32)
        for c in range(3):
            glow[:, :, c] = self.blur_channel(neon[:, :, c].astype(np.float32), kernel_1d, k)

        result = neon.astype(np.float32) + glow * 0.7
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
