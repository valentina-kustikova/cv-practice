from abc import ABC, abstractmethod 

import numpy as np


def to_gray(image):
    """Яркость Y = 0.299 R + 0.587 G + 0.114 B"""
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def convolve(channel, kernel):
    """Свертка одноканального изображения с ядром нечетного размера."""
    k = kernel.shape[0] #ядро
    pad = k // 2 #отступы с краев
    padded = np.pad(channel.astype(np.float64), pad, mode='edge') 
    h, w = channel.shape
    result = np.zeros((h, w))
    for i in range(k):
        for j in range(k):
            result += kernel[i, j] * padded[i:i + h, j:j + w] #берем слепок изображения и двигаем его. умножаем каждый пиксель слепка на ячейку ядра.
    return result


def to_uint8(image):
    """Ограничивает значения диапазоном яркости [0, 255] и приводит массив к uint8."""
    return np.clip(image, 0, 255).astype(np.uint8)


class ImageFilter(ABC):
    @staticmethod 
    def get_filter(args):
        filters = {
            'resize': lambda: Resize(args.width, args.height, args.scale),
            'gray': lambda: RGB2GrayScale(),
            'antique': lambda: Antique(),
            'fade': lambda: FadeColor(args.strength),
            'film': lambda: InfraredFilm(args.grain, args.seed),
            'matte': lambda: Matte(args.radius, args.softness),
            'old': lambda: OldPhoto(args.noise, args.scratches, args.seed),
            'neon': lambda: Neon(args.threshold, args.color),
        }
        if args.filter not in filters:
            raise ValueError(f'Неизвестный фильтр: {args.filter}')
        return filters[args.filter]() 

    @abstractmethod
    def apply_filter(self, image):
        pass


class Resize(ImageFilter):
    """Масштабирование изображения с использованием билинейной интерполяции"""
    def __init__(self, width=None, height=None, scale=None):
        self.width = width
        self.height = height
        self.scale = scale

    def apply_filter(self, image):
        h, w = image.shape[:2]
        if self.scale is not None:
            new_w, new_h = round(w * self.scale), round(h * self.scale)
        elif self.width is not None and self.height is not None:
            new_w, new_h = self.width, self.height
        else:
            raise ValueError('Для resize задайте --scale или --width и --height')
        if new_w <= 0 or new_h <= 0:
            raise ValueError('Размер изображения должен быть положительным')

        # Координаты центров новых пикселей в исходном изображении
        ys = np.clip((np.arange(new_h) + 0.5) * h / new_h - 0.5, 0, h - 1) 
        xs = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5, 0, w - 1)  

        #берем соседей сверху слева и сверху справа. снизу слева и снизу справа.
        y0, x0 = ys.astype(int), xs.astype(int)
        y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)

        # Веса соседей: дробная часть координаты. 
        dy = (ys - y0)[:, None, None]
        dx = (xs - x0)[None, :, None]

        img = image.astype(np.float64)
        top = img[y0][:, x0] * (1 - dx) + img[y0][:, x1] * dx 
        bottom = img[y1][:, x0] * (1 - dx) + img[y1][:, x1] * dx 
        return to_uint8(np.round(top * (1 - dy) + bottom * dy)) 


class RGB2GrayScale(ImageFilter):
    """Фильтр преобразования полноцветного изображения в ч/б."""
    def apply_filter(self, image):
        return to_uint8(to_gray(image))


class Antique(ImageFilter): #Сепия
    # Матрица сепии для порядка каналов BGR
    SEPIA = np.array([[0.131, 0.534, 0.272],
                      [0.168, 0.686, 0.349],
                      [0.189, 0.769, 0.393]])

    def apply_filter(self, image):
        return to_uint8(image @ self.SEPIA.T)


class FadeColor(ImageFilter):
    """Эффект выцветшей фотографии"""
    def __init__(self, strength=0.5):
        if not 0 <= strength <= 1:
            raise ValueError('strength должен быть от 0 до 1')
        self.strength = strength

    def apply_filter(self, image):
        s = self.strength
        img = image.astype(np.float64)
        gray = to_gray(img)[:, :, None] 
        desaturated = img + s * (gray - img) #обесцвечивание
        low, high = 70 * s, 255 - 40 * s #падение контраста.
        return to_uint8(low + desaturated * (high - low) / 255) #переводим значения из шкалы [0,255] в шкалу [low,high]


class InfraredFilm(ImageFilter):
    """Имитация инфракрасной черно-белой фотопленки"""
    def __init__(self, grain=12.0, seed=None):
        self.grain = grain
        self.seed = seed

    def apply_filter(self, image):
        img = image.astype(np.float64)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        ir = np.clip(-0.7 * r + 2.0 * g - 0.3 * b, 0, 255) # Канальный микшер ИК-пленки.
        glow = convolve(ir, np.ones((7, 7)) / 49) # Ореол вокруг светлых участков
        ir = 0.7 * ir + 0.3 * np.maximum(ir, glow)
        # Зерно пленки
        rng = np.random.default_rng(self.seed)
        ir += rng.normal(0, self.grain, ir.shape)
        return to_uint8(ir)


class Matte(ImageFilter):
    """Эффект виньетки "Матте" """
    def __init__(self, radius=0.7, softness=0.3): #softness - плавность перехода
        if softness <= 0:
            raise ValueError('softness должен быть больше 0')
        self.radius = radius
        self.softness = softness

    def apply_filter(self, image):
        h, w = image.shape[:2]
        cy, cx = h / 2, w / 2
        y, x = np.mgrid[0:h, 0:w] + 0.5  # координаты центров пикселей
        d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)  
        alpha = np.clip((d - self.radius) / self.softness, 0, 1)[:, :, None] 
        return to_uint8(image * (1 - alpha) + 255 * alpha) #блендинг


class OldPhoto(ImageFilter):
    """Эффект состаренной фотографии"""
    def __init__(self, noise=20.0, scratches=30, seed=None):
        self.noise = noise
        self.scratches = scratches
        self.seed = seed

    def apply_filter(self, image):
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]
        img = image.astype(np.float64)
        img += rng.normal(0, self.noise, (h, w))[:, :, None]

        # Царапины
        thickness = max(1, w // 800) # толщина царапины
        for _ in range(self.scratches): # создаем царапины
            x = rng.integers(0, w)
            y0 = rng.integers(0, h) #начало царапины
            y1 = y0 + rng.integers(h // 5, h)  # ее конец
            img[y0:y1, x:x + thickness] = rng.choice([230, 30]) # рисуем саму царапину
        return to_uint8(img)


class Neon(ImageFilter):
    """Неоновый эффект"""
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    def __init__(self, threshold=0.2, color=(255, 0, 255)):
        self.threshold = threshold 
        self.color = np.array(color[::-1])  # RGB -> BGR

    def apply_filter(self, image):
        gray = convolve(to_gray(image), np.ones((3, 3)) / 9) 
        
        #градиенты яркости
        gx = convolve(gray, self.SOBEL_X)
        gy = convolve(gray, self.SOBEL_X.T)
        magnitude = np.sqrt(gx ** 2 + gy ** 2) #считаем длину вектора градиента 
        if magnitude.max() > 0: 
            magnitude /= magnitude.max()

        edges = (magnitude > self.threshold).astype(np.float64) 
        glow = convolve(edges, np.ones((9, 9)) / 81) #свечение контура
        light = np.clip(edges + 2 * glow, 0, 1) 
        return to_uint8(light[:, :, None] * self.color)