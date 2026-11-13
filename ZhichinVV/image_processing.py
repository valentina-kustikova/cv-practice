import abc
import numpy as np
import cv2
from dataclasses import dataclass


class ImageFilter(abc.ABC):

    @staticmethod
    def get_filter(image, params):
        filter = params['filter']

        if filter == "resize":
            return Resize(image, params)
        if filter == "gray":
            return Gray(image, params)
        if filter == "antique":
            return Antique(image, params)
        if filter == "fade":
            return Fade(image, params)
        if filter == "kodak_aerochrome":
            return Kodak_aerochrome(image, params)
        if filter == "matte":
            return Matte(image, params)
        if filter == "scratches":
            return Scratches(image, params)
        if filter == "neon":
            return Neon(image, params)

        
        raise ValueError(f"filter '{filter}' not found")

    @abc.abstractmethod
    def apply_filter(self):
        raise ValueError(f'Not implemented')


@dataclass
class Resize(ImageFilter):
    image: np.ndarray
    params: dict

    def __post_init__(self):
        if self.params['scale'] <= 0:
            raise ValueError(f'scale cannot have such a meaning')

    def apply_filter(self):

        scale = self.params['scale']
        img = self.image

        if scale > 1:
            kernel = np.ones((3, 3), np.float32) / 9
            img = cv2.filter2D(img, -1, kernel)

        h, w = img.shape[:2]
        new_h = max(1, int(h / scale))
        new_w = max(1, int(w / scale))

        ys = np.clip((np.arange(new_h) + 0.5) * h / new_h - 0.5, 0, h - 1)
        xs = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5, 0, w - 1)

        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)

        wy = ys - y0
        wx = xs - x0
        wy = wy[:, None, None]
        wx = wx[None, :, None]

        f = img.astype(np.float32)
        up = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
        down = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
        res = up * (1 - wy) + down * wy

        return np.clip(res, 0, 255).astype(np.uint8)


@dataclass
class Gray(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        b = self.image[:, :, 0].astype(np.float32)
        g = self.image[:, :, 1].astype(np.float32)
        r = self.image[:, :, 2].astype(np.float32)
        gray_image = np.clip(b*0.114 + g*0.587 + r*0.299, 0, 255).astype(np.uint8)

        return gray_image


@dataclass
class Antique(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        kernel = np.array( 
            [
                [0.131, 0.534, 0.272],
                [0.168, 0.686, 0.349],
                [0.189, 0.769, 0.393]
            ], dtype=np.float32
        )
        sepia = self.image.astype(np.float32).dot(np.transpose(kernel))

        return np.clip(sepia, 0, 255).astype(np.uint8)


@dataclass
class Fade(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        rng = np.random.default_rng()
        img_float = self.image.astype(np.float32)

        brightness = img_float.mean(axis=2)
        shape = (*brightness.shape, 1)

        conditions = [
            brightness < 64,                          
            (brightness >= 64) & (brightness < 128),  
            (brightness >= 128) & (brightness < 192), 
            brightness >= 192                         
        ]
        conditions = [cond[..., np.newaxis] for cond in conditions]

        choice = [
            img_float + rng.integers(25, 41, size=shape),   
            img_float + rng.integers(15, 26, size=shape),   
            img_float + rng.integers(8, 16, size=shape),   
            img_float - rng.integers(5, 13, size=shape),   
        ]

        fade = np.select(conditions, choice, default=img_float)
        return np.clip(fade, 0, 255).astype(np.uint8)


@dataclass
class Kodak_aerochrome(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        kernel = np.array([
            [ 0.9,  0.0,  0.1],   
            [ 0.2, -0.2,  0.5],  
            [-0.4,  1,  0.1]
        ], dtype=np.float32)

        out = self.image.astype(np.float32).dot(kernel.T)
        return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Matte(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        h, w = self.image.shape[:2]
        cx, cy = w//2, h//2  
        a, b =  (w*3)//8, (h*3)//8
        y, x = np.mgrid[:h, :w]

        ellipse_ = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) > 1
        self.image[ellipse_] = 255

        return np.clip(self.image, 0, 255).astype(np.uint8)


@dataclass
class Scratches(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        rng = np.random.default_rng()
        img = self.image.astype(np.float32)
        h, w = img.shape[:2]

        img += rng.normal(0, self.params['noise'], img.shape)

        layer = np.zeros((h, w), np.uint8)
        for _ in range(15):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            length = int(rng.integers(h // 10, h // 2))
            dx = int(rng.integers(-length // 10, length // 10 + 1))
            cv2.line(layer, (x, y), (x + dx, y + length), 255, 1)

        img[layer > 0] += 60

        return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class Neon(ImageFilter):
    image: np.ndarray
    params: dict

    def apply_filter(self):
        edges = cv2.Canny(self.image, 100, 200)

        color = np.array((255, 255, 0), np.float32)
        neon = (edges[..., np.newaxis] / 255.0) * color

        glow = cv2.GaussianBlur(neon, (0, 0), 8)

        base = self.image.astype(np.float32)
        base *= 0.3
        res = base + neon + glow 

        return np.clip(res, 0, 255).astype(np.uint8)