from abc import ABC, abstractmethod

import numpy as np


def to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def luminance(image_f: np.ndarray) -> np.ndarray:
    if image_f.ndim == 2:
        return image_f
    b, g, r = image_f[..., 0], image_f[..., 1], image_f[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(channel, ((py, py), (px, px)), mode="edge")
    h, w = channel.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            coef = kernel[i, j]
            if coef != 0:
                out += coef * padded[i:i + h, j:j + w]
    return out


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    k = gaussian_kernel_1d(sigma)
    row_kernel = k[None, :]
    col_kernel = k[:, None]

    def blur_channel(ch):
        return convolve2d(convolve2d(ch, row_kernel), col_kernel)

    if image.ndim == 2:
        return blur_channel(image.astype(np.float32))
    return np.stack([blur_channel(image[..., c].astype(np.float32))
                     for c in range(image.shape[2])], axis=2)


def hsv_to_bgr(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    def f(n):
        k = (n + h * 6.0) % 6.0
        return v - v * s * np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)
    return np.stack([f(1.0), f(3.0), f(5.0)], axis=-1)


def parse_color(color: str) -> np.ndarray:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Некорректный цвет '{color}', ожидается #RRGGBB")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return np.array([b, g, r], dtype=np.float32) / 255.0


class ImageFilter(ABC):

    _registry: dict = {}

    name: str = ""

    @classmethod
    def register(cls, filter_cls):
        cls._registry[filter_cls.name] = filter_cls
        return filter_cls

    @staticmethod
    def get_filter(name: str, **params) -> "ImageFilter":
        try:
            filter_cls = ImageFilter._registry[name]
        except KeyError:
            available = ", ".join(sorted(ImageFilter._registry))
            raise ValueError(f"Неизвестный фильтр '{name}'. Доступны: {available}")
        return filter_cls(**params)

    @staticmethod
    def available_filters():
        return sorted(ImageFilter._registry)

    @abstractmethod
    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        pass



@ImageFilter.register
class Resize(ImageFilter):

    name = "resize"

    def __init__(self, width=None, height=None, scale=None,
                 interpolation="bilinear"):
        self.width, self.height, self.scale = width, height, scale
        if interpolation not in ("nearest", "bilinear"):
            raise ValueError("interpolation должен быть 'nearest' или 'bilinear'")
        self.interpolation = interpolation

    def _target_size(self, h, w):
        if self.scale is not None:
            if self.scale <= 0:
                raise ValueError("scale должен быть > 0")
            return max(1, round(h * self.scale)), max(1, round(w * self.scale))
        if self.width is None and self.height is None:
            raise ValueError("Укажите width/height или scale")
        new_w = self.width if self.width else round(w * self.height / h)
        new_h = self.height if self.height else round(h * self.width / w)
        if new_w <= 0 or new_h <= 0:
            raise ValueError("Размеры должны быть положительными")
        return int(new_h), int(new_w)

    def apply_filter(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self._target_size(h, w)

        ys = (np.arange(new_h, dtype=np.float32) + 0.5) * (h / new_h) - 0.5
        xs = (np.arange(new_w, dtype=np.float32) + 0.5) * (w / new_w) - 0.5

        if self.interpolation == "nearest":
            yi = np.clip(np.round(ys).astype(int), 0, h - 1)
            xi = np.clip(np.round(xs).astype(int), 0, w - 1)
            return image[yi[:, None], xi[None, :]]

        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        dy = (ys - y0)[:, None]
        dx = (xs - x0)[None, :]
        if image.ndim == 3:
            dy, dx = dy[..., None], dx[..., None]

        img = image.astype(np.float32)
        top = img[y0[:, None], x0[None, :]] * (1 - dx) + img[y0[:, None], x1[None, :]] * dx
        bottom = img[y1[:, None], x0[None, :]] * (1 - dx) + img[y1[:, None], x1[None, :]] * dx
        out = top * (1 - dy) + bottom * dy
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)



@ImageFilter.register
class RGB2GrayScale(ImageFilter):
    name = "grayscale"

    def apply_filter(self, image):
        if image.ndim == 2:
            return image.copy()
        return to_uint8(luminance(to_float(ensure_bgr(image))))



@ImageFilter.register
class Antique(ImageFilter):

    name = "antique"

    # матрица сепии для вектора (R, G, B)
    SEPIA = np.array([[0.393, 0.769, 0.189],
                      [0.349, 0.686, 0.168],
                      [0.272, 0.534, 0.131]], dtype=np.float32)

    def __init__(self, intensity=1.0, contrast=0.85, vignette=0.18):
        if not 0 <= intensity <= 1:
            raise ValueError("intensity должен быть в [0, 1]")
        if not 0 <= vignette <= 1:
            raise ValueError("vignette должен быть в [0, 1]")
        self.intensity = intensity
        self.contrast = contrast
        self.vignette = vignette

    def apply_filter(self, image):
        img = to_float(ensure_bgr(image))
        rgb = img[..., ::-1]                         
        sepia_rgb = rgb @ self.SEPIA.T               
        sepia = np.clip(sepia_rgb[..., ::-1], 0, 1)  
        out = (1 - self.intensity) * img + self.intensity * sepia
        out = (out - 0.5) * self.contrast + 0.5      
        h, w = out.shape[:2]
        y = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / (h / 2)
        x = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / (w / 2)
        distance = np.sqrt((x[None, :] ** 2 + y[:, None] ** 2) / 2)
        t = np.clip((distance - 0.3) / 0.7, 0, 1)
        shade = 1 - self.vignette * t * t * (3 - 2 * t)
        return to_uint8(out * shade[..., None])



@ImageFilter.register
class FadeColor(ImageFilter):

    name = "fade"

    def __init__(self, strength=0.6):
        if not 0 <= strength <= 1:
            raise ValueError("strength должен быть в [0, 1]")
        self.strength = strength

    def apply_filter(self, image):
        k = self.strength
        img = to_float(ensure_bgr(image))
        y = luminance(img)[..., None]
        out = img + 0.7 * k * (y - img)              
        lo, hi = 0.18 * k, 1 - 0.10 * k               
        out = lo + (hi - lo) * out                   
        tint = np.array([-0.04, 0.0, 0.03], np.float32) * k   
        return to_uint8(out + tint)



@ImageFilter.register
class InfraredFilm(ImageFilter):

    name = "film"

    def __init__(self, mode="bw", halation=0.35, grain=0.04, seed=None):
        if mode not in ("bw", "color"):
            raise ValueError("mode должен быть 'bw' или 'color'")
        self.mode, self.halation, self.grain = mode, halation, grain
        self.rng = np.random.default_rng(seed)

    def apply_filter(self, image):
        img = to_float(ensure_bgr(image))
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        ir = np.clip(0.2 * r + 1.4 * g - 0.6 * b, 0, 1)

        if self.mode == "bw":
            out = np.repeat(ir[..., None], 3, axis=2)
        else:
            out = np.stack([g, r, ir], axis=2)

        if self.halation > 0:
            h, w = out.shape[:2]
            sigma = max(h, w) / 150.0
            bright = np.clip(out - 0.6, 0, 1) / 0.4
            glow = gaussian_blur(bright, sigma)
            out = 1 - (1 - out) * (1 - self.halation * glow)

        if self.grain > 0:
            noise = self.rng.normal(0, self.grain, out.shape[:2]).astype(np.float32)
            out = out + noise[..., None]
        return to_uint8(out)



@ImageFilter.register
class Matte(ImageFilter):

    name = "matte"

    def __init__(self, radius=0.85, feather=0.25, color="#FFFFFF"):
        if radius <= 0 or feather < 0:
            raise ValueError("radius > 0, feather >= 0")
        self.radius, self.feather = radius, feather
        self.color = parse_color(color)

    def apply_filter(self, image):
        img = to_float(ensure_bgr(image))
        h, w = img.shape[:2]
        y = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / (h / 2)
        x = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / (w / 2)
        d = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2) / self.radius
        if self.feather > 0:
            t = np.clip((d - 1) / self.feather, 0, 1)
            alpha = t * t * (3 - 2 * t)               
        else:
            alpha = (d > 1).astype(np.float32)
        alpha = alpha[..., None]
        out = img * (1 - alpha) + self.color * alpha
        return to_uint8(out)



@ImageFilter.register
class OldPhoto(ImageFilter):
    name = "old"

    def __init__(self, scratches=15, dust=300, noise=0.06,
                 sepia=0.8, vignette=0.5, seed=None):
        self.scratches, self.dust, self.noise = scratches, dust, noise
        self.sepia, self.vignette = sepia, vignette
        self.rng = np.random.default_rng(seed)

    def _scratches(self, h, w):
        mask = np.zeros((h, w), np.float32)
        for _ in range(self.scratches):
            length = max(2.0, min(h, w) * self.rng.uniform(0.025, 0.075))
            steps = max(4, int(np.ceil(length / 0.5)))
            angles = self.rng.uniform(0, 2 * np.pi) + np.cumsum(
                self.rng.normal(0, 0.18, steps))
            xs = self.rng.uniform(0, w) + np.cumsum(0.5 * np.cos(angles))
            ys = self.rng.uniform(0, h) + np.cumsum(0.5 * np.sin(angles))
            x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
            dx, dy = xs - x0, ys - y0
            strength = self.rng.uniform(0.2, 0.45)
            taper = np.sin(np.linspace(0, np.pi, steps)) ** 0.5
            for ox, oy, weight in ((0, 0, (1 - dx) * (1 - dy)),
                                   (1, 0, dx * (1 - dy)),
                                   (0, 1, (1 - dx) * dy),
                                   (1, 1, dx * dy)):
                xi, yi = x0 + ox, y0 + oy
                valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                np.maximum.at(mask, (yi[valid], xi[valid]),
                              (strength * taper * weight)[valid])
        return mask

    def _dust(self, h, w):
        mask = np.zeros((h, w), np.float32)
        for _ in range(self.dust):
            cy, cx = self.rng.integers(0, h), self.rng.integers(0, w)
            r = self.rng.uniform(0.5, 2.5)
            ri = int(np.ceil(r))
            y0, y1 = max(0, cy - ri), min(h, cy + ri + 1)
            x0, x1 = max(0, cx - ri), min(w, cx + ri + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            mask[y0:y1, x0:x1][disk] = self.rng.uniform(0.4, 1.0)
        return mask

    def apply_filter(self, image):
        base = ensure_bgr(image)
        h, w = base.shape[:2]
        out = to_float(Antique(intensity=self.sepia, contrast=0.9,
                               vignette=0).apply_filter(base))

        y = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / (h / 2)
        x = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / (w / 2)
        d2 = (x[None, :] ** 2 + y[:, None] ** 2) / 2
        out = out * (1 - self.vignette * d2)[..., None]

        if self.noise > 0:
            out = out + self.rng.normal(0, self.noise, (h, w)).astype(np.float32)[..., None]

        scratch = self._scratches(h, w)[..., None]
        out = out * (1 - scratch) + 0.95 * scratch
        dust = self._dust(h, w)[..., None]
        out = out * (1 - dust) + 0.08 * dust
        return to_uint8(out)



@ImageFilter.register
class Neon(ImageFilter):

    name = "neon"

    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    SOBEL_Y = SOBEL_X.T
    HALO = ((2, 1.8), (7, 1.2), (20, 0.65))

    def __init__(self, color="#BF00FF", threshold=0.3, glow=1.0,
                 intensity=1.0, background=0.7):
        if not 0 < threshold < 1:
            raise ValueError("threshold должен быть в (0, 1)")
        if glow < 0 or intensity < 0:
            raise ValueError("glow и intensity должны быть >= 0")
        if not 0 <= background <= 1:
            raise ValueError("background должен быть в [0, 1]")
        self.color = color if color == "rainbow" else parse_color(color)
        self.threshold, self.glow = threshold, glow
        self.intensity, self.background = intensity, background

    @staticmethod
    def _thin_edges(magnitude, gx, gy):
        h, w = magnitude.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        norm = np.maximum(np.sqrt(gx ** 2 + gy ** 2), 1e-8)
        dx, dy = gx / norm, gy / norm

        def sample(x, y):
            x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
            fx, fy = x - x0, y - y0
            return (magnitude[y0, x0] * (1 - fx) * (1 - fy)
                    + magnitude[y0, x1] * fx * (1 - fy)
                    + magnitude[y1, x0] * (1 - fx) * fy
                    + magnitude[y1, x1] * fx * fy)

        forward = sample(xx + dx, yy + dy)
        backward = sample(xx - dx, yy - dy)
        keep = (magnitude > forward) & (magnitude >= backward)
        return magnitude * keep

    def apply_filter(self, image):
        img = to_float(ensure_bgr(image))
        scale = max(min(img.shape[:2]) / 1080.0, 0.35)
        gray = gaussian_blur(luminance(img), max(1.0, 2.0 * scale))
        gx = convolve2d(gray, self.SOBEL_X)
        gy = convolve2d(gray, self.SOBEL_Y)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = mag / max(np.percentile(mag, 99), 1e-6)
        mag = self._thin_edges(mag, gx, gy)
        upper = min(1.5 * self.threshold, 1)
        t = np.clip((mag - self.threshold) / (upper - self.threshold), 0, 1)
        edges = t * t * (3 - 2 * t)

        if isinstance(self.color, str):            
            hue = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
            color = hsv_to_bgr(hue, np.ones_like(hue), np.ones_like(hue))
        else:
            color = self.color

        halo = sum(weight * sigma * scale * gaussian_blur(edges, sigma * scale)
                   for sigma, weight in self.HALO)
        core = gaussian_blur(edges, max(0.4, 0.6 * scale))
        light = (self.glow * halo[..., None] * color
                 + 3.5 * core[..., None] * (0.85 + 0.15 * color))
        light = 1 - np.exp(-self.intensity * light)
        base = img * self.background
        out = 1 - (1 - base) * (1 - light)
        return to_uint8(out)
