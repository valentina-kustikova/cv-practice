from abc import ABC, abstractmethod
import numpy as np

class ImageFilter(ABC):
    @staticmethod
    def get_filter(name, **kwargs):
        if name == "resize":
            return Resize(**kwargs)
        elif name == "gray":
            return RGB2GrayScale(**kwargs)
        elif name == "antique":
            return Antique(**kwargs)
        elif name == "fade":
            return FadeColor(**kwargs)
        elif name == "film":
            return Film(**kwargs)
        elif name == "matte":
            return Matte(**kwargs)
        elif name == "scratches":
            return Scratches(**kwargs)
        elif name == "neon":                    # ← новая ветка
            return Neon(**kwargs)
        else:
            raise ValueError(f"Неизвестный фильтр: {name}")

    @abstractmethod
    def apply_filter(self, image):
        pass
    
    

class RGB2GrayScale(ImageFilter):

    def apply_filter(self, image):
        
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        return gray
    
    def __init__(self, scale=0.5, **kwargs):
        pass

class Resize(ImageFilter):

    def __init__(self, scale=0.5, **kwargs):
        self.scale = scale

    def apply_filter(self, image):
        h, w = image.shape[:2]

        new_h = int(h * self.scale)
        new_w = int(w * self.scale)

        row_indices = (np.arange(new_h) / self.scale).astype(np.int32)
        col_indices = (np.arange(new_w) / self.scale).astype(np.int32)


        result = image[row_indices[:, None], col_indices]

        return result


class Antique(ImageFilter):

    def __init__(self, intensity=1.0, **kwargs):

        self.intensity = max(0.0, min(1.0, intensity))  # ограничиваем [0, 1]

    def apply_filter(self, image):

        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]


        r_new = 0.393 * r + 0.769 * g + 0.189 * b
        g_new = 0.349 * r + 0.686 * g + 0.168 * b
        b_new = 0.272 * r + 0.534 * g + 0.131 * b


        sepia = np.stack([b_new, g_new, r_new], axis=2)

 
        result = img * (1.0 - self.intensity) + sepia * self.intensity

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class FadeColor(ImageFilter):

    def __init__(self, intensity=0.5, **kwargs):
        self.intensity = max(0.0, min(1.0, intensity))

        # цвет «пыли»
        self.fade_color = np.array([200, 220, 240], dtype=np.float32)

    def apply_filter(self, image):

        img = image.astype(np.float32)

        # Параметры эффекта на основе intensity:
        contrast = 1.0 - self.intensity * 0.7
        fade = self.intensity * 0.5               

        #    128 — средний серый
        img = 128.0 + (img - 128.0) * contrast

        img = img * (1.0 - fade) + self.fade_color * fade

        result = np.clip(img, 0, 255).astype(np.uint8)

        return result


class Film(ImageFilter):
    # инфрокрасное

    def __init__(self, intensity=1.0, **kwargs):

        self.intensity = max(0.0, min(1.0, intensity))

    def apply_filter(self, image):
        img = image.astype(np.float32)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        k = self.intensity
        r_new = r * (1.0 + 0.30 * k)     
        g_new = g * (1.0 - 0.30 * k)     
        b_new = b * (1.0 - 0.50 * k)     

        result = np.stack([b_new, g_new, r_new], axis=2)

        # нормальное распределение для шума
        sigma = 12.0 * k
        noise = np.random.normal(0.0, sigma, result.shape).astype(np.float32)
        result = result + noise

        # немножко уменьшим контраст
        contrast = 1.0 - 0.1 * k
        result = 128.0 + (result - 128.0) * contrast


        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class Matte(ImageFilter):

    def __init__(self, intensity=1, border=0.15, **kwargs):
        
        self.border = max(0.0, min(0.5, border))
        self.intensity = max(0.0, min(1.0, intensity))

    def apply_filter(self, image):

        h, w = image.shape[:2]


        cx = w / 2.0
        cy = h / 2.0


        # border — доля, которую «отрезаем» от края.

        a = cx * (1.0 - self.border)
        b = cy * (1.0 - self.border)

        # экономный способ создания массивов
        Y, X = np.ogrid[:h, :w]

        dist = np.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2)

        mask = 1.0 - np.clip(dist, 0.0, 1.0)


        #    mask[:, :, None] — добавляем третью ось для трёх каналов.
        img = image.astype(np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)

        result = img * mask[:, :, None] + white * (1.0 - mask[:, :, None])

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class Scratches(ImageFilter):


    def __init__(self, intensity=1.0, count=None, **kwargs):

        self.intensity = max(0.0, min(1.0, intensity))
        # count - число царапин
        if count is None:
            self.count = int(10 * self.intensity)   # до 10 царапин
        else:
            self.count = max(0, int(count))

    def apply_filter(self, image):
        img = image.astype(np.float32).copy()
        h, w = img.shape[:2]

        for _ in range(self.count):

            x = np.random.randint(0, w)

            y_start = np.random.randint(0, h // 2)
            y_end = np.random.randint(h // 2, h)

            if np.random.rand() < 0.7:
                color = 255.0
            else:
                color = 0.0

            # толщина 1 или 2 пикселя.
            thickness = np.random.randint(1, 3)

            # x_min и x_max — границы по горизонтали
            x_min = max(0, x - thickness)
            x_max = min(w, x + thickness + 1)

            img[y_start:y_end, x_min:x_max] = color

        # шум
        sigma = 8.0 * self.intensity
        noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
        img = img + noise

        img = np.clip(img, 0, 255)

        # смешиваем с оригиналом по intensity.
        original = image.astype(np.float32)
        result = original * (1.0 - self.intensity) + img * self.intensity

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


class Neon(ImageFilter):

    def __init__(self, intensity=1.0, threshold=50, **kwargs):
        self.intensity = max(0.0, min(1.0, intensity))
        self.threshold = max(0, min(255, threshold))
        self.neon_color = np.array([255, 0, 255], dtype=np.float32)

    def apply_filter(self, image):
        # контуры ищем на одном канале
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        kernel_blur = np.ones((3, 3)) / 9
        gray = self._convolve(gray, kernel_blur)


        Gx = self._convolve(gray, np.array([[-1, 0, 1],
                                             [-2, 0, 2],
                                             [-1, 0, 1]], dtype=np.float32))

        Gy = self._convolve(gray, np.array([[-1, -2, -1],
                                             [ 0,  0,  0],
                                             [ 1,  2,  1]], dtype=np.float32))

        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        edges = (magnitude > self.threshold).astype(np.float32)

        h, w = gray.shape
        # Фон - затемнённая версия оригинала
        background = (image.astype(np.float32) * 0.15)

        # Накладываем контуры
        result = background + edges[:, :, None] * self.neon_color * self.intensity

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def _convolve(image, kernel):
        h, w = image.shape
        result = np.zeros((h, w), dtype=np.float32)

        for ky in range(3):
            for kx in range(3):
                result[1:h-1, 1:w-1] += (
                    kernel[ky, kx] *
                    image[ky:h-2+ky, kx:w-2+kx]
                )

        return result