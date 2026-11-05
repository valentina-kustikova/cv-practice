from abc import ABC, abstractmethod

import numpy as np


class ImageFilter(ABC):
    # Базовый класс для всех фильтров

    @staticmethod
    def get_filter(name, param=None):
        # Создаёт фильтр по его имени
        if name == "gray":
            return GrayScale()
        if name == "resize":
            return Resize(param)
        if name == "antique":
            return Antique(param)
        if name == "fade_color":
            return FadeColor(param)
        if name == "film_effect":
            return FilmEffect(param)
        if name == "matte":
            return Matte(param)
        if name == "old_photo":
            return OldPhoto(param)
        if name == "neon":
            return Neon(param)

        raise ValueError(f"Неизвестный фильтр: {name}")

    @abstractmethod
    def apply_filter(self, image):
        # Применяет фильтр к изображению
        pass


class GrayScale(ImageFilter):
    # Перевод в оттенки серого

    def apply_filter(self, image):
        img = image.astype(np.float32)

        # каналы в порядке OpenCV: B, G, R
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # яркость как взвешенная сумма каналов
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        return np.clip(gray, 0, 255).astype(np.uint8)


class Resize(ImageFilter):
    # Изменение разрешения методом ближайшего соседа

    def __init__(self, scale=None):
        if scale is None:
            scale = 0.5
        if scale <= 0:
            raise ValueError("Масштаб должен быть больше 0")
        self.scale = scale

    def apply_filter(self, image):
        height, width = image.shape[:2]

        # размеры нового изображения
        new_height = int(height * self.scale)
        new_width = int(width * self.scale)

        # номера строк и столбцов исходного изображения для каждого нового пикселя
        rows = (np.arange(new_height) / self.scale).astype(int)
        cols = (np.arange(new_width) / self.scale).astype(int)

        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)

        # выбираем нужные строки, затем нужные столбцы
        return image[rows][:, cols]


class Antique(ImageFilter):
    # Эффект антиквариата (сепия)

    def __init__(self, param=None):
        if param is None:
            param = 1.0
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # каждый новый канал — взвешенная сумма всех трёх исходных
        new_r = 0.393 * r + 0.769 * g + 0.189 * b
        new_g = 0.349 * r + 0.686 * g + 0.168 * b
        new_b = 0.272 * r + 0.534 * g + 0.131 * b

        sepia = np.stack([new_b, new_g, new_r], axis=2)

        # смешиваем сепию с оригиналом
        result = (1 - self.param) * img + self.param * sepia

        return np.clip(result, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    # Выцветание

    def __init__(self, param=None):
        if param is None:
            param = 0.5
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        # смешиваем изображение с белым цветом
        faded = img * (1 - self.param) + 255 * self.param

        return np.clip(faded, 0, 255).astype(np.uint8)


class FilmEffect(ImageFilter):
    # Имитация инфракрасной плёнки

    def __init__(self, param=None):
        if param is None:
            param = 1.0
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        # меняем местами красный и зелёный каналы
        infrared = np.stack([b, r, g], axis=2)

        # смешиваем результат с оригиналом
        result = (1 - self.param) * img + self.param * infrared

        return np.clip(result, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    # Белая овальная рамка по краям

    def __init__(self, param=None):
        if param is None:
            param = 0.6
        if not (0 <= param < 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1)")
        self.param = param

    def apply_filter(self, image):
        img = image.astype(np.float32)
        height, width = img.shape[:2]

        # центр изображения
        cx = width / 2
        cy = height / 2

        # координаты всех столбцов и строк
        x = np.arange(width)
        y = np.arange(height)

        # расстояние каждого пикселя до центра в долях полуосей
        dx = (x[None, :] - cx) / (width / 2)
        dy = (y[:, None] - cy) / (height / 2)

        d = np.sqrt(dx ** 2 + dy ** 2)

        # маска: 1 в центре, плавно до 0 к краям
        mask = (1 - d) / (1 - self.param)
        mask = np.clip(mask, 0, 1)
        mask = mask[:, :, None]

        # смешиваем изображение с белым по маске
        result = img * mask + 255 * (1 - mask)

        return np.clip(result, 0, 255).astype(np.uint8)


def box_blur(channel):
    # Размытие усреднением по окну 3x3
    p = np.pad(channel, 1, mode="edge")
    return (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]) / 9


class OldPhoto(ImageFilter):
    # Состаренная фотография

    def __init__(self, param=None):
        if param is None:
            param = 0.5
        if not (0 <= param <= 1):
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        self.param = param

    def apply_filter(self, image):
        # пожелтение через сепию
        img = Antique(0.8).apply_filter(image).astype(np.float32)
        height, width = img.shape[:2]

        # затемнение краёв
        x = np.arange(width)
        y = np.arange(height)
        dx = (x[None, :] - width / 2) / (width / 2)
        dy = (y[:, None] - height / 2) / (height / 2)
        d = np.sqrt(dx ** 2 + dy ** 2)
        darken = np.clip(1 - 0.5 * d ** 2, 0, 1)
        img = img * darken[:, :, None]

        # зерно плёнки
        sigma = 30 * self.param
        noise = np.random.normal(0, sigma, (height, width, 1))
        img = img + noise

        # царапины разной длины, яркости и наклона
        scratches = int(30 * self.param)
        for _ in range(scratches):
            x0 = np.random.randint(0, width)
            y0 = np.random.randint(0, height)
            length = np.random.randint(height // 5, height)
            slope = np.random.uniform(-0.1, 0.1)
            brightness = np.random.uniform(40, 120)

            rows = np.arange(y0, min(y0 + length, height))
            cols = (x0 + slope * (rows - y0)).astype(int)
            cols = np.clip(cols, 0, width - 1)
            img[rows, cols] = img[rows, cols] + brightness

        # пыль: светлые и тёмные точки
        dust = int(600 * self.param)
        ys = np.random.randint(0, height, dust)
        xs = np.random.randint(0, width, dust)
        half = dust // 2
        img[ys[:half], xs[:half]] = 230
        img[ys[half:], xs[half:]] = 30

        return np.clip(img, 0, 255).astype(np.uint8)


class Neon(ImageFilter):
    # Неоновое свечение контуров

    def __init__(self, param=None):
        if param is None:
            param = 0.3
        if not (0 < param < 1):
            raise ValueError("Порог должен быть в диапазоне (0, 1)")
        self.threshold = param

    def apply_filter(self, image):
        img = image.astype(np.float32)

        # яркость и сглаживание
        gray = GrayScale().apply_filter(image).astype(np.float32)
        gray = box_blur(gray)

        # оператор Собеля
        p = np.pad(gray, 1, mode="edge")
        gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - \
             (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - \
             (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        edges = np.sqrt(gx ** 2 + gy ** 2)
        edges = edges / edges.max()

        # порог: оставляем только сильные контуры
        lines = (edges > self.threshold).astype(np.float32)

        # свечение: размываем линии несколько раз
        glow = lines
        for _ in range(4):
            glow = box_blur(glow)
        light = np.clip(lines + 1.2 * glow, 0, 1)[:, :, None]

        # яркий цвет контура и тёмный фон
        color = img / (img.max(axis=2, keepdims=True) + 1) * 255
        result = img * 0.15 + color * light

        return np.clip(result, 0, 255).astype(np.uint8)
