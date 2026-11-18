from abc import ABC, abstractmethod
import numpy as np


class ImageFilter(ABC):
    @staticmethod
    def get_filter(filter_name, **params):
        # По названию выбираем класс фильтра, а **params передаёт
        # его конструктору только те настройки, которые ему нужны.
        filters_map = {
            "resize": Resize,
            "gray": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "film": Film,
            "matte": Matte,
            "scratches": ScratchesNoise,
            "neon": Neon,
        }
        if filter_name not in filters_map:
            raise ValueError(f"Фильтр '{filter_name}' не найден")
        return filters_map[filter_name](**params)

    @abstractmethod
    def apply_filter(self, image):
        # Каждый наследник должен определить собственную обработку изображения.
        pass


class Resize(ImageFilter):
    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Размеры должны быть положительными")
        self.w = int(width)
        self.h = int(height)

    def apply_filter(self, image):
        # Берём исходные размеры изображения (высота и ширина).
        src_h, src_w = image.shape[:2]
        
        # Для каждого пикселя результата находим соответствующую координату
        # в исходном изображении. Координаты могут быть дробными.
        y_points = np.linspace(0, src_h - 1, self.h)
        x_points = np.linspace(0, src_w - 1, self.w)
        
        # Индексы четырёх соседних пикселей: слева/справа и сверху/снизу.
        # np.minimum не даёт выйти за правую и нижнюю границы изображения.
        y0 = np.floor(y_points).astype(int)
        x0 = np.floor(x_points).astype(int)
        y1 = np.minimum(y0 + 1, src_h - 1)
        x1 = np.minimum(x0 + 1, src_w - 1)
        
        # Дробные части координат задают вес соседних пикселей при смешивании.
        dy = (y_points - y0)
        dx = (x_points - x0)
        
        # Добавляем измерения, чтобы веса автоматически применялись
        # ко всем цветовым каналам через broadcasting NumPy.
        if image.ndim == 3:
            dy = dy[:, np.newaxis, np.newaxis]
            dx = dx[np.newaxis, :, np.newaxis]
        else:
            dy = dy[:, np.newaxis]
            dx = dx[np.newaxis, :]
            
        # Сначала интерполируем по горизонтали для верхнего и нижнего рядов,
        # затем смешиваем полученные строки по вертикали.
        row0 = image[y0[:, None], x0] * (1 - dx) + image[y0[:, None], x1] * dx
        row1 = image[y1[:, None], x0] * (1 - dx) + image[y1[:, None], x1] * dx
        
        res = row0 * (1 - dy) + row1 * dy
        return np.clip(res, 0, 255).astype(np.uint8)


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        # Веса учитывают вклад цветовых каналов в воспринимаемую яркость.
        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        gray = image.astype(np.float32) @ weights
        return np.clip(gray, 0, 255).astype(np.uint8)



class Antique(ImageFilter):
    def __init__(self, intensity=0.85, vignette=0.55):
        self.alpha = max(0.0, min(1.0, float(intensity)))
        self.vignette = max(0.0, min(1.0, float(vignette)))

    def apply_filter(self, image):
        # float32 нужен для матричных вычислений и дробных промежуточных значений.
        img = image.astype(np.float32)

        sepia_mat = np.array([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393]
        ], dtype=np.float32)

        # Преобразуем цветовые каналы матрицей сепии и смешиваем
        # полученный результат с исходным изображением.
        sepia_img = img @ sepia_mat.T
        res = img * (1.0 - self.alpha) + sepia_img * self.alpha

        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx = w / 2.0
        cy = h / 2.0

        # Расстояние от центра используется для построения виньетки:
        # чем дальше пиксель от центра, тем сильнее затемнение.
        dist = np.sqrt(
            ((x - cx) / cx) ** 2 +
            ((y - cy) / cy) ** 2
        )

        # Значения маски лежат в диапазоне 0..1 и умножаются на все каналы.
        mask = np.clip(
            1.0 - self.vignette * dist ** 2,
            0.0,
            1.0
        )

        if img.ndim == 3:
            mask = mask[..., np.newaxis]

        res = res * mask

        return np.clip(res, 0, 255).astype(np.uint8)



class FadeColor(ImageFilter):
    def __init__(self, strength=0.35):
        self.fade = max(0.0, min(1.0, float(strength)))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        
        # Смешиваем изображение с белым цветом.
        mixed = img * (1.0 - self.fade) + 255.0 * self.fade
        # Среднее по каналам для каждого пикселя; keepdims сохраняет
        # размерность канала, чтобы NumPy мог выполнить broadcasting.
        mean_val = mixed.mean(axis=2, keepdims=True)
        
        # Сдвигаем каждый канал к среднему значению: насыщенность становится ниже.
        res = mean_val + (mixed - mean_val) * (1.0 - self.fade * 0.5)
        return np.clip(res, 0, 255).astype(np.uint8)


class Film(ImageFilter):
    def __init__(self, grain=15.0, tint=0.4):
        self.grain = float(grain)
        self.tint = max(0.0, min(1.0, float(tint)))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        
        color_mat = np.array([
            [0.2, 0.7, 0.1],
            [0.4, 0.4, 0.2],
            [0.9, 0.05, 0.05]
        ], dtype=np.float32)
        
        # Матрица меняет соотношение каналов, создавая необычный оттенок.
        infrared = img @ color_mat.T
        res = img * (1.0 - self.tint) + infrared * self.tint
        
        # Гауссов шум добавляет мелкую случайную зернистость.
        if self.grain > 0:
            noise = np.random.normal(0, self.grain, res.shape)
            res += noise
            
        return np.clip(res, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def __init__(self, radius=0.75, softness=0.2):
        self.rad = max(0.1, min(1.0, float(radius)))
        self.soft = max(0.01, float(softness))

    def apply_filter(self, image):
        img = image.astype(np.float32)
        h, w = img.shape[:2]
        
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        
        # Нормированное расстояние до центра эллипса.
        d = np.sqrt(((x - cx) / (cx * self.rad)) ** 2 + ((y - cy) / (cy * self.rad)) ** 2)
        
        # Получаем маску перехода и сглаживаем её функцией smoothstep.
        weight = np.clip((d - 1.0 + self.soft) / self.soft, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        
        if img.ndim == 3:
            weight = weight[..., np.newaxis]
            
        # По маске смешиваем исходные пиксели с белым цветом.
        res = img * (1.0 - weight) + 255.0 * weight
        return np.clip(res, 0, 255).astype(np.uint8)

class ScratchesNoise(ImageFilter):
    def __init__(self, noise_level=15.0, scratch_count=20, scratch_intensity=0.7):
        if noise_level < 0:
            raise ValueError("Уровень шума не может быть отрицательным")
        if scratch_count < 0:
            raise ValueError("Количество царапин не может быть отрицательным")
        if not 0 <= scratch_intensity <= 1:
            raise ValueError("Интенсивность царапин должна быть в диапазоне [0, 1]")

        self.noise_level = float(noise_level)
        self.scratch_count = int(scratch_count)
        self.scratch_intensity = float(scratch_intensity)

    def apply_filter(self, image):
        img = image.astype(np.float32)
        height, width = img.shape[:2]

        # Добавляем гауссов шум с нулевым средним.
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, (height, width, 1))
            img += noise

        # Создаём маску, в которой будут отмечены царапины.
        scratches = np.zeros((height, width), dtype=np.float32)

        for _ in range(self.scratch_count):
            # Выбираем случайную начальную точку и длину царапины.
            x0 = np.random.randint(0, width)
            y0 = np.random.randint(0, height)
            min_length = max(1, height // 20)
            max_length = max(min_length + 1, height // 3)
            length = np.random.randint(min_length, max_length + 1)

            # Небольшой наклон делает царапины менее одинаковыми.
            slope = np.random.uniform(-0.08, 0.08)
            rows = np.arange(y0, min(y0 + length, height))
            cols = (x0 + slope * (rows - y0)).astype(int)
            cols = np.clip(cols, 0, width - 1)

            # Записываем яркость царапины в маску.
            strength = np.random.uniform(0.2, 0.45) * self.scratch_intensity
            scratches[rows, cols] = strength

        # Смешиваем пиксели царапин со светло-серым цветом.
        scratches = scratches[:, :, np.newaxis]
        img = img * (1.0 - scratches) + 220.0 * scratches

        # Добавляем случайные светлые и тёмные точки, похожие на пыль.
        dust_count = max(1, int(height * width * 0.0005))
        dust_y = np.random.randint(0, height, dust_count)
        dust_x = np.random.randint(0, width, dust_count)
        half = dust_count // 2
        img[dust_y[:half], dust_x[:half]] = 230
        img[dust_y[half:], dust_x[half:]] = 30

        return np.clip(img, 0, 255).astype(np.uint8)


class Neon(ImageFilter):
    def __init__(self, threshold=40, glow=1.5, sigma=4.0):
        self.thr = float(threshold)
        self.glow = float(glow)
        self.sigma = float(sigma)

        if self.thr < 0:
            raise ValueError("Порог не может быть отрицательным")
        if self.glow < 0:
            raise ValueError("Интенсивность свечения не может быть отрицательной")
        if self.sigma <= 0:
            raise ValueError("Sigma должна быть положительной")

    def gaussian_blur(self, image, sigma=None):
        # sigma определяет ширину размытия: меньшее значение даёт узкую кайму.
        sigma = self.sigma if sigma is None else max(0.5, float(sigma))
        radius = max(1, int(3 * sigma))
        x = np.arange(-radius, radius + 1)

        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        padded = np.pad(image, ((0, 0), (radius, radius), (0, 0)), mode="edge")

        blurred = np.zeros_like(image, dtype=np.float32)

        for i, weight in enumerate(kernel):
            blurred += weight * padded[:, i:i + image.shape[1], :]

        padded = np.pad(blurred, ((radius, radius), (0, 0), (0, 0)), mode="edge")

        result = np.zeros_like(image, dtype=np.float32)

        for i, weight in enumerate(kernel):
            result += weight * padded[i:i + image.shape[0], :, :]

        return result

    def apply_filter(self, image):
        img = image.astype(np.float32)

        weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        gray = img @ weights

        # Вычисляем горизонтальную и вертикальную составляющие градиента.
        padded = np.pad(gray, 1, mode="edge")

        gx = (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]
        ) - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])

        gy = (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        ) - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])

        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        edges = (magnitude > self.thr).astype(np.float32)

        contours = np.zeros_like(img)

        contours[..., 0] = edges * 255
        contours[..., 1] = edges * 80
        contours[..., 2] = edges * 255

        # Небольшое размытие смягчает границы контура.
        inner_glow = self.gaussian_blur(contours, sigma=max(0.5, self.sigma * 0.5))

        # Более широкое размытие делает цветной контур заметным вокруг края.
        outer_glow = self.gaussian_blur(contours, sigma=self.sigma)

        # Сохраняем исходное изображение и добавляем к нему цветные контуры.
        result = (img + inner_glow * self.glow * 0.8 + outer_glow * self.glow * 2.0 + contours * 0.9)

        return np.clip(result, 0, 255).astype(np.uint8)