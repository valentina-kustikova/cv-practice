import numpy as np
from PIL import Image

def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def get_gray(img):
    if img.ndim == 2:
        return img.astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return img.astype(np.float32) @ weights


def convolve(img, kernel):
    h, w = img.shape
    size = kernel.shape[0] // 2
    padded = np.pad(img, size, mode='edge')
    result = np.zeros((h, w), dtype=np.float32)

    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            result += kernel[y, x] * padded[y:y + h, x:x + w]

    return result


class ImageFilter:
    @staticmethod
    def get_filter(name, **params):
        name = name.lower()
        if name == 'resize':
            return Resize(params['new_h'], params['new_w'])
        if name == 'gray':
            return RGB2GrayScale()
        if name == 'antique':
            return Antique()
        if name == 'fade':
            return FadeColor(params.get('alpha', 0.5),
                             params.get('fade_color', (200, 200, 200)))
        if name == 'film':
            return Film(params.get('grain_sigma', 10), params.get('seed'))
        if name == 'matte':
            return Matte(
                params.get('radius', 1.25),
                params.get('feather', 0.4)
            )
        if name == 'neon':
            return Neon(params.get('sob_threshold', 50))
        if name == 'scratches_texture':
            return ScratchesTexture(params.get('noise_sigma', 15),
                                    params.get('texture_alpha', 0.65),
                                    params.get('seed'))
        raise ValueError('Неизвестный фильтр: ' + name)

    def apply_filter(self, img):
        raise NotImplementedError


class Resize(ImageFilter):
    def __init__(self, new_h, new_w):
        if new_h <= 0 or new_w <= 0:
            raise ValueError('Новый размер должен быть больше нуля')
        self.new_h = new_h
        self.new_w = new_w

    def apply_filter(self, img):
        h, w = img.shape[:2]
        y, x = np.meshgrid(np.arange(self.new_h), np.arange(self.new_w),
                           indexing='ij')
        old_y = np.floor(y * h / self.new_h).astype(np.int32)
        old_x = np.floor(x * w / self.new_w).astype(np.int32)
        old_y = np.clip(old_y, 0, h - 1)
        old_x = np.clip(old_x, 0, w - 1)
        return img[old_y, old_x].copy()


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, img):
        return to_uint8(get_gray(img))


class Antique(ImageFilter):
    def apply_filter(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Для сепии нужно RGB-изображение')
        matrix = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]], dtype=np.float32)
        result = img.astype(np.float32) @ matrix.T
        h, w = img.shape[:2]
        cx, cy = (w - 1) / 2, (h - 1) / 2
        x = (np.arange(w, dtype=np.float32) - cx) / max(cx, 0.5)
        y = (np.arange(h, dtype=np.float32) - cy) / max(cy, 0.5)
        distance_squared = (y[:, None] ** 2 + x[None, :] ** 2) / 2
        vignette_strength = 0.75
        vignette = 1 - vignette_strength * distance_squared
        result *= vignette[..., None]
        return to_uint8(result)


class FadeColor(ImageFilter):
    def __init__(self, alpha=0.5, fade_color=(200, 200, 200)):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha должен быть от 0 до 1')
        if len(fade_color) != 3 or any(color < 0 or color > 255
                                       for color in fade_color):
            raise ValueError('fade_color содержит три значения от 0 до 255')
        self.alpha = alpha
        self.fade_color = np.array(fade_color, dtype=np.float32)

    def apply_filter(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Для выцветания нужно RGB-изображение')
        result = (1 - self.alpha) * img.astype(np.float32)
        result += self.alpha * self.fade_color
        return to_uint8(result)


class Film(ImageFilter):
    def __init__(self, grain_sigma=10, seed=None):
        if grain_sigma < 0:
            raise ValueError('grain_sigma не может быть отрицательным')
        self.grain_sigma = grain_sigma
        self.seed = seed

    def apply_filter(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Для плёнки нужно RGB-изображение')
        source = img.astype(np.float32)
        aerochrome_matrix = np.array([
            [ -0.5,  4.5, -2.3 ], 
            [ 1.0,  -2.0,  2.0 ], 
            [ 0.0, -0.6,  1.6 ]   
        ])
        result = np.dot(source, aerochrome_matrix.T)
        
        result = np.clip(result, 0, 255)
        
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, self.grain_sigma, result.shape)
        result = np.clip(result + noise, 0, 255)
        
        return to_uint8(result)


class Matte(ImageFilter):
    def __init__(self, radius=1.25, feather=0.4):
        if radius <= 0 or feather <= 0:
            raise ValueError('Радиус и плавность должны быть больше нуля')

        self.radius = radius
        self.feather = feather

    def apply_filter(self, img):
        h, w = img.shape[:2]
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2
        radius_x = max(center_x, 0.5)
        radius_y = max(center_y, 0.5)
        distance = np.sqrt(((x - center_x) / radius_x) ** 2 +
                           ((y - center_y) / radius_y) ** 2)
        weight = np.clip((self.radius - distance) / self.feather, 0, 1)
        if img.ndim == 3:
            weight = weight[:, :, None]
        result = weight * img.astype(np.float32) + (1 - weight) * 255
        return to_uint8(result)

class ScratchesTexture(ImageFilter):
    def __init__(self, noise_sigma=15, texture_alpha=0.65, seed=None):
        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError('noise_sigma должен быть конечным числом не меньше нуля')
        if not 0 <= texture_alpha <= 1:
            raise ValueError('texture_alpha должен быть от 0 до 1')
        self.noise_sigma = noise_sigma
        self.texture_alpha = texture_alpha
        self.seed = seed

    def apply_filter(self, img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError('Для состаривания нужно RGB-изображение')
        h, w = img.shape[:2]

        matrix = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]], dtype=np.float32)
        result = np.clip(img.astype(np.float32) @ matrix.T, 0, 255)

        rng = np.random.default_rng(self.seed)
        result = np.clip(result + rng.normal(0, self.noise_sigma, (h, w, 1)), 0, 255)

        with Image.open('images/tekstyra_small.jpg') as source:
            texture = np.array(source)

        texture = Resize(h, w).apply_filter(texture)
        if texture.ndim == 3 and texture.shape[2] == 4:
            texture = (texture[..., :3].astype(np.float32) * (texture[..., 3:4] / 255))
        texture = get_gray(texture)
        mask = texture[..., None] / 255

        screen = 255 - (255 - result) * (1 - mask)
        return to_uint8((1 - self.texture_alpha) * result + self.texture_alpha * screen)


class Neon(ImageFilter):
    def __init__(self, sob_threshold=50):
        if sob_threshold < 0:
            raise ValueError('Порог Собеля не может быть отрицательным')
        self.sob_threshold = sob_threshold

    def apply_filter(self, img):
        gray = get_gray(img)
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=np.float32)
        gradient_x = convolve(gray, kernel_x)
        gradient_y = convolve(gray, kernel_y)
        magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        edges = (magnitude > self.sob_threshold).astype(np.float32)
        glow = convolve(edges, np.ones((5, 5), dtype=np.float32) / 25)
        strength = np.clip(edges + 2 * glow, 0, 1)
        halo = strength.copy()
        kernel = np.ones((7, 7), dtype=np.float32) / 49
        for _ in range(45):
            halo = convolve(halo, kernel)
        strength = np.maximum(strength, 1.2 * halo)
        color = np.array([0, 255, 255], dtype=np.float32)
        return to_uint8(strength[:, :, None] * color)
