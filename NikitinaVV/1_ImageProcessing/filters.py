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


def size_scale(image):
    """Масштаб изображения относительно Full HD.
    Нужен, чтобы размеры эффектов в пикселях не зависели от разрешения."""
    return min(image.shape[:2]) / 1080


def smoothstep(edge0, edge1, x):
    """Плавный переход Эрмита от 0 (x <= edge0) к 1 (x >= edge1) по кривой 3t^2 - 2t^3."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def radial_distance(h, w):
    """Нормированное расстояние от центра кадра до центра пикселя: 0 в центре, 1 на вписанном эллипсе, sqrt(2) в углах."""
    y, x = np.ogrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    return np.sqrt(((x + 0.5 - cx) / cx) ** 2 + ((y + 0.5 - cy) / cy) ** 2)


def gaussian_blur(image, sigma):
    """Гауссово размытие."""
    radius = max(1, int(np.ceil(3 * sigma))) 
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2)) 
    kernel /= kernel.sum() #сумма весов равна 1

    result = image.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * result.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, mode='edge') 
        n = result.shape[axis] #длина изображения вдоль текущей оси
        blurred = np.zeros_like(result)
        window = [slice(None)] * result.ndim
        for i, weight in enumerate(kernel):
            window[axis] = slice(i, i + n)  # копия изображения, сдвинутая на i - radius
            blurred += weight * padded[tuple(window)]
        result = blurred
    return result


def draw_spot(mask, cx, cy, radius, opacity):
    """Рисует на маске мягкий круг с центром (cx, cy). opacity(непрозрачность пятна): 0 - невидимое, 1 - ярко белое"""
    h, w = mask.shape
    r = int(np.ceil(radius)) + 1 
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1) #координаты квадрата
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1: 
        return
    y, x = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2) #расстояния от точек внутри квадрата до центра пылинки
    spot = opacity * (1 - smoothstep(radius - 0.5, radius + 0.5, dist)) 
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], spot)

class ImageFilter(ABC):
    @staticmethod 
    def get_filter(args):
        filters = {
            'resize': lambda: Resize(args.width, args.height, args.scale),
            'gray': lambda: RGB2GrayScale(),
            'antique': lambda: Antique(args.vignette),
            'fade': lambda: FadeColor(args.strength),
            'film': lambda: InfraredFilm(args.grain, args.seed),
            'matte': lambda: Matte(args.radius, args.softness),
            'old': lambda: OldPhoto(args.noise, args.dust, args.hairs, args.seed),
            'neon': lambda: Neon(args.threshold, args.color, args.glow, args.background),
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


class Antique(ImageFilter):
    """Сепия с темной виньеткой по краям"""
    SEPIA = np.array([[0.131, 0.534, 0.272],
                      [0.168, 0.686, 0.349],
                      [0.189, 0.769, 0.393]])

    def __init__(self, vignette=0.6):
        if not 0 <= vignette <= 1:
            raise ValueError('vignette должен быть от 0 до 1')
        self.vignette = vignette  # насколько затемняются углы кадра

    def apply_filter(self, image):
        h, w = image.shape[:2]
        sepia = image @ self.SEPIA.T
        d = radial_distance(h, w) / np.sqrt(2)  # расстояние до углов. 0 в центре, 1 в углах
        shade = 1 - self.vignette * smoothstep(0.3, 1, d)
        return to_uint8(sepia * shade[:, :, None])


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
    """Имитация цветной инфракрасной пленки: зеленые и теплые тона
    становятся красно-розовыми, небо и вода - голубыми"""
    SKY = np.array([[1.0, 0.0, 0.0],      # B' = B
                    [0.5, 0.5, 0.0],      # G' = 0.5G + 0.5B
                    [-0.15, 0.0, 1.15]])  # R' = 1.15R - 0.15B
    PINK_GAMMA = np.array([1.3, 2.2, 0.35])

    def __init__(self, grain=8.0, seed=None):
        if grain < 0:
            raise ValueError('grain не может быть отрицательным')
        self.grain = grain
        self.seed = seed

    def apply_filter(self, image):
        s = size_scale(image)
        h, w = image.shape[:2]
        img = image.astype(np.float64)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Маска областей, которые сильно отражают ИК-свет: зеленые и теплые тона, где max(R, G) > B.
        # Небо и вода (B > R, G) в маску не попадают.
        warmth = (np.maximum(r, g) - b) / 255
        mask = gaussian_blur(smoothstep(0, 0.12, warmth), 1.5 * s)[:, :, None] 

        sky = img @ self.SKY.T 
        y = to_gray(img) / 255
       
        pink = 255 * (y ** 0.8)[:, :, None] ** self.PINK_GAMMA  
        result = sky * (1 - mask) + pink * mask

        glow = gaussian_blur(result, 6 * s) 
        result += 0.3 * np.maximum(glow - result, 0)

        y = to_gray(result) / 255
        weight = np.clip(1 - 4 * (y - 0.5) ** 2, 0.1, 1) 
        rng = np.random.default_rng(self.seed)
        result += (rng.normal(0, self.grain, (h, w)) * weight)[:, :, None]
        return to_uint8(result)


class Matte(ImageFilter):
    """Эффект виньетки "Матте": белая овальная рамка вокруг центральной части кадра"""
    def __init__(self, radius=0.7, softness=0.3):  # softness - ширина перехода к белому
        if not 0 < radius < 1:
            raise ValueError('radius должен быть больше 0 и меньше 1, иначе овал не помещается в кадр')
        if softness <= 0:
            raise ValueError('softness должен быть больше 0')
        self.radius = radius
        self.softness = softness

    def apply_filter(self, image):
        h, w = image.shape[:2]
        d = radial_distance(h, w)
        outer = min(self.radius + self.softness, 1) # внешняя граница, где переход должен стать 100% чисто белым.
        alpha = smoothstep(self.radius, outer, d)[:, :, None]
        return to_uint8(image * (1 - alpha) + 255 * alpha)  # блендинг с белым


class OldPhoto(ImageFilter):
    """Эффект старой пленочной фотографии: выцветшие теплые цвета, неравномерная засветка,
    зерно, белые пылинки и ворсинки"""
    TINT = np.array([0.88, 0.97, 1.05]) 

    def __init__(self, noise=7.0, dust=900, hairs=8, seed=None):
        if noise < 0 or dust < 0 or hairs < 0:
            raise ValueError('noise, dust и hairs не могут быть отрицательными')
        self.noise = noise
        self.dust = dust
        self.hairs = hairs
        self.seed = seed

    def apply_filter(self, image):
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]
        s = size_scale(image)
        img = image.astype(np.float64)

        # 1. Выцветание
        gray = to_gray(img)[:, :, None]
        img = (gray + 0.6 * (img - gray)) * self.TINT 
        img = 25 + img * (235 - 25) / 255 
        img = gaussian_blur(img, 0.6 * s) 

        # 2. Неравномерная засветка: сумма косинусных волн низкой частоты (до 2 периодов на кадр)
        y, x = np.ogrid[0:h, 0:w]
        light = np.zeros((h, w))
        for _ in range(6):
            fx, fy = rng.uniform(-2, 2, 2) # пространственные частоты
            light += np.cos(2 * np.pi * (fx * x / w + fy * y / h) + rng.uniform(0, 2 * np.pi))
        light /= np.abs(light).max() # нормировка в [-1;1]
        d = radial_distance(h, w) / np.sqrt(2)
        shade = (1 + 0.1 * light) * (1 - 0.35 * smoothstep(0.4, 1, d))  # волновые пятна и легкая виньетка
        img *= shade[:, :, None]

        # 3. Белые пылинки разного размера  
        specks = np.zeros((h, w))
        for _ in range(self.dust):
            radius = s * (0.6 + 2.4 * rng.random() ** 3)
            draw_spot(specks, rng.uniform(0, w), rng.uniform(0, h), radius, rng.uniform(0.4, 1))

        # 4. Ворсинки. Направление нити меняется случайно на каждом шаге.
        step = 0.5
        for _ in range(self.hairs):
            n = int(s * rng.uniform(30, 110) / step) # количество точек в нити
            angle = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.12, n)) # изгиб нити. начальное направление + накопленные повороты
            xs = rng.uniform(0, w) + np.cumsum(step * np.cos(angle))
            ys = rng.uniform(0, h) + np.cumsum(step * np.sin(angle))
            radius, opacity = s * rng.uniform(0.5, 0.9), rng.uniform(0.5, 0.9)
            for cx, cy in zip(xs, ys):
                draw_spot(specks, cx, cy, radius, opacity)

        img = img * (1 - specks[:, :, None]) + 240 * specks[:, :, None]

        # 5. Зерно пленки
        grain = gaussian_blur(rng.normal(0, 1, (h, w)), 0.7 * s)
        img += (self.noise / grain.std() * grain)[:, :, None]
        return to_uint8(img)


class Neon(ImageFilter):
    """Неоновый эффект: контуры светятся заданным цветом, вокруг них мягкий ореол"""
    SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    HALO = ((2, 0.8), (6, 1.0), (16, 1.5))

    def __init__(self, threshold=0.25, color=(255, 0, 255), glow=1.0, background=0.15):
        if not 0 < threshold < 1:
            raise ValueError('threshold должен быть больше 0 и меньше 1')
        if glow < 0:
            raise ValueError('glow не может быть отрицательным')
        if not 0 <= background <= 1:
            raise ValueError('background должен быть от 0 до 1')
        self.threshold = threshold
        self.color = np.array(color[::-1]) / 255  
        self.glow = glow
        self.background = background

    def apply_filter(self, image):
        s = size_scale(image)
        gray = gaussian_blur(to_gray(image), 1.2 * s)  # сглаживание убирает шумовые контуры

        # Градиенты яркости и их модуль
        gx = convolve(gray, self.SOBEL_X) 
        gy = convolve(gray, self.SOBEL_X.T)
        magnitude = np.sqrt(gx ** 2 + gy ** 2) 
        magnitude = np.clip(magnitude / max(np.percentile(magnitude, 99), 1e-6), 0, 1)
        edges = smoothstep(self.threshold, min(1.5 * self.threshold, 1), magnitude) 

        # Свечение
        halo = sum(weight * gaussian_blur(edges, sigma * s) for sigma, weight in self.HALO)
        light = (edges + self.glow * halo)[:, :, None] * self.color + 0.8 * edges[:, :, None]
        light = 1 - np.exp(-light)
        return to_uint8(self.background * image + 255 * light)
