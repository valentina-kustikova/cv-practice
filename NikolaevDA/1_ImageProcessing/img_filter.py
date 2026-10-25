from abc import ABC, abstractmethod
import numpy as np

def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)

def to_gray(image):
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    gray_image = image @ weights
    return gray_image

class ImageFilter(ABC):
    @staticmethod
    def get_filter(args):
        filters = {
            'resize': lambda: Resize(args.width, args.height),
            'gray': lambda: RGB2GrayScale(),
            'antique': lambda: Antique(),
            'fade': lambda: FadeColor(args.fade),
            'film': lambda: Film(args.grain, args.seed),
            'matte': lambda: Matte(args.radius, args.softness),
            'old': lambda: OldPhoto(args.intensity, args.scratches, args.seed),
            'neon': lambda: Neon(),
        }
        if args.filter not in filters:
            raise ValueError(f'Неизвестный фильтр: {args.filter}')
        return filters[args.filter]()

    @abstractmethod
    def apply_filter(self, image):
        pass

class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        return to_uint8(to_gray(image))

class Resize:
    def __init__(self, width, height):
            if width is None or height is None:
                raise ValueError("для фильтра resize нужно указать --width и --height")
            if width <= 0 or height <= 0:
                raise ValueError("ширина и высота должны быть положительными")
            self.width = width
            self.height = height

    def apply_filter(self, image):
        src_h, src_w = image.shape[:2]
        dst_h, dst_w = self.height, self.width

        y = (np.arange(dst_h) * (src_h / dst_h)).astype(np.int32)
        x = (np.arange(dst_w) * (src_w / dst_w)).astype(np.int32)

        y = np.clip(y, 0, src_h - 1)
        x = np.clip(x, 0, src_w - 1)

        return image[np.ix_(y, x)]

class Antique:
    SEPIA = np.array([[0.131, 0.534, 0.272],
                      [0.168, 0.686, 0.349],
                      [0.189, 0.769, 0.393]])

    def apply_filter(self, image):
        return to_uint8(image @ self.SEPIA.T)

class FadeColor:
    def __init__(self, strength=0.5):
        if not 0 <= strength <= 1:
            raise ValueError('strength должен быть от 0 до 1')
        self.strength = strength

    def apply_filter(self, image):
        s = self.strength
        img = image.astype(np.float64)
        gray = to_gray(img)[:, :, None] 
        desaturated = img + s * (gray - img)
        low, high = 70 * s, 255 - 40 * s
        return to_uint8(low + desaturated * (high - low) / 255)

class Film:
    def __init__(self, grain=12.0, seed=None):
            self.grain = grain
            self.seed = seed
    
    def apply_filter(self, image):
        lift=0.10
        warmth=0.12
        img = image.astype(np.float32)

        # lift black point
        k = lift
        img = img * (1.0 - k) + k * 255.0

        # warmth
        w = warmth
        img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + w),       0, 255)  # R
        img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + w * 0.5), 0, 255)  # G
        img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - w * 0.5), 0, 255)  # B

        if self.grain > 0:
            rng = np.random.default_rng(self.seed)
            noise = rng.normal(0.0, self.grain, img.shape[:2])[:, :, None]
            img = img + noise

        return to_uint8(img)

class Matte:
    def __init__(self, radius=0.7, softness=0.3):
        if softness <= 0:
            raise ValueError('softness cant be <= 0')
        self.radius = radius
        self.softness = softness

    def apply_filter(self, image):
        h, w = image.shape[:2]
        cy, cx = h / 2, w / 2
        y, x = np.mgrid[0:h, 0:w] + 0.5
        d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)  
        alpha = np.clip((d - self.radius) / self.softness, 0, 1)[:, :, None] 
        return to_uint8(image * (1 - alpha) + 255 * alpha)

class OldPhoto:
    def __init__(self, intensity=30.0, scratches=30, seed=None):
        self.intensity = intensity
        self.scratches = scratches
        self.seed = seed

    def apply_filter(self, image):
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]
        img = image.astype(np.float64)
        img += rng.normal(0, self.intensity, (h, w))[:, :, None]
        thickness = max(1, w // 800)
        for _ in range(self.scratches):
            x = rng.integers(0, w)
            y0 = rng.integers(0, h)
            y1 = y0 + rng.integers(h // 5, h)
            img[y0:y1, x:x+thickness] = rng.choice([230, 30])
        return to_uint8(img)

class Neon:
    def apply_filter(self, image):
        img_gray = image.astype(np.float32)
        if len(img_gray.shape) == 3:
            img_gray = to_gray(image)

        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = kernel_x.T

        h, w = img_gray.shape
        padded = np.pad(img_gray, 1, mode='edge')
        
        gx = np.zeros_like(img_gray)
        gy = np.zeros_like(img_gray)

        for i in range(3):
            for j in range(3):
                gx += padded[i:i+h, j:j+w] * kernel_x[i, j]
                gy += padded[i:i+h, j:j+w] * kernel_y[i, j]

        magnitude = np.sqrt(gx**2 + gy**2)
        edges = to_uint8(magnitude)

        neon = np.zeros((*edges.shape, 3), dtype=np.uint8)
        neon[:, :, 0] = np.clip(edges * 1.0, 0, 255).astype(np.uint8)
        neon[:, :, 1] = np.clip(edges * 0.2, 0, 255).astype(np.uint8)
        neon[:, :, 2] = np.clip(edges * 1.5, 0, 255).astype(np.uint8)
        return neon