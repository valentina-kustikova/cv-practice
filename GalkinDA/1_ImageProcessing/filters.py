import numpy as np

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
            return Matte()
        if name == 'neon':
            return Neon(params.get('sob_threshold', 50))
        if name == 'scratches':
            return ScratchesNoise(params.get('noise_sigma', 15),
                                  params.get('num_scratches', 7),
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
            [ 0.0,  2.0, -1.0 ], 
            [ 1.0,  0.0,  0.0 ], 
            [ 0.0, -1.0,  2.0 ]   
        ])
        result = np.dot(source, aerochrome_matrix.T)
        
        result = np.clip(result, 0, 255)
        
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, self.grain_sigma, result.shape)
        result = np.clip(result + noise, 0, 255)
        
        return to_uint8(result)


class Matte(ImageFilter):
    def apply_filter(self, img):
        h, w = img.shape[:2]
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2
        radius_x = max(center_x, 0.5)
        radius_y = max(center_y, 0.5)
        distance = np.sqrt(((x - center_x) / radius_x) ** 2 +
                           ((y - center_y) / radius_y) ** 2)
        weight = np.clip((1 - distance) / 0.5, 0, 1)
        if img.ndim == 3:
            weight = weight[:, :, None]
        result = weight * img.astype(np.float32) + (1 - weight) * 255
        return to_uint8(result)

class ScratchesNoise(ImageFilter):
    def __init__(self, noise_sigma=15, num_scratches=7, seed=None):
        if not isinstance(noise_sigma, (int, float, np.integer, np.floating)) or \
                not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError('noise_sigma должен быть конечным числом не меньше нуля')
        if not isinstance(num_scratches, (int, np.integer)) or num_scratches < 0:
            raise ValueError('num_scratches должен быть целым числом не меньше нуля')
        self.noise_sigma = noise_sigma
        self.num_scratches = num_scratches
        self.seed = seed

    def apply_filter(self, img):
        rng = np.random.default_rng(self.seed)

        result = img.astype(np.float32)
        result += rng.normal(0, self.noise_sigma, img.shape)

        h, w = img.shape[:2]
        for _ in range(self.num_scratches):
            x0, x1 = rng.integers(0, w, size=2)
            y0, y1 = rng.integers(0, h, size=2)
            thickness = rng.integers(1, 3)
            dx, dy = x1 - x0, y1 - y0
            n = max(abs(dx), abs(dy)) + 1
            t = np.linspace(0, 1, n)
            x = np.rint(x0 + t * dx).astype(int)[:, None]
            y = np.rint(y0 + t * dy).astype(int)[:, None]

            offsets = np.arange(thickness)[None, :]
            if abs(dx) >= abs(dy):
                x = x + np.zeros_like(offsets)
                y = y + offsets
            else:
                x = x + offsets
                y = y + np.zeros_like(offsets)

            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            result[y[valid], x[valid]] = 255

        return to_uint8(result)

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
        color = np.array([0, 255, 255], dtype=np.float32)
        return to_uint8(strength[:, :, None] * color)
