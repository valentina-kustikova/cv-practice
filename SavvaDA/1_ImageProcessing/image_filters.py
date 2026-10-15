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

    def __init__(self, intensity=1.0, border=0.15, **kwargs):
        self.border = max(0.0, min(0.5, border))
        self.intensity = max(0.0, min(1.0, intensity))

    def apply_filter(self, image):
        if self.intensity == 0:
            return image.copy()

        h, w = image.shape[:2]
        Y, X = np.ogrid[:h, :w]

        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        rx = max(cx * (1.0 - self.border), 1.0)
        ry = max(cy * (1.0 - self.border), 1.0)

        # Нормированное расстояние: 0 в центре, 1 — на границе овала.
        dist = np.sqrt(
            ((X - cx) / rx) ** 2 +
            ((Y - cy) / ry) ** 2
        )

        mask = np.clip((dist - 0.85) / 0.15, 0.0, 1.0)

        # smoothstep — сглаживает кривую перехода.
        mask = mask * mask * (3.0 - 2.0 * mask)

        # Всё, что ЗА границей овала — принудительно белое.
        mask = np.where(dist >= 1.0, 1.0, mask).astype(np.float32)

        img = image.astype(np.float32)
        alpha = (mask * self.intensity)[:, :, None]

        result = img * (1.0 - alpha) + 255.0 * alpha

        return np.clip(result, 0, 255).astype(np.uint8)





class Scratches(ImageFilter):

    def __init__(self, intensity=1.0, count=None, **kwargs):
        self.intensity = max(0.0, min(1.0, intensity))

        if count is None:
            self.count = int(250 * self.intensity)
        else:
            self.count = max(0, int(count))
            
    def _blur_mask(self, mask):
        h, w = mask.shape
        padded = np.pad(mask, 1, mode="edge")
        result = np.zeros((h, w), dtype=np.float32)

        # Ядро 3x3 с повышенным весом центрального пикселя.
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)

        kernel /= kernel.sum()

        for dy in range(3):
            for dx in range(3):
                result += (
                    padded[dy:dy + h, dx:dx + w]
                    * kernel[dy, dx]
                )

        return result

    def apply_filter(self, image):
        if self.intensity == 0:
            return image.copy()

        img = image.astype(np.float32).copy()
        h, w = img.shape[:2]

        if h < 2 or w < 2:
            return image.copy()

        # Слой царапин и карта их прозрачности
        scratches = img.copy()
        alpha_map = np.zeros((h, w), dtype=np.float32)

        for _ in range(self.count):
            # Разные длины: от коротких потёртостей
            # до длинных царапин.
            length = np.random.randint(2, max(3, int(h * 0.05) + 1)
)

            x0 = np.random.randint(0, w)
            y0 = np.random.randint(0, h)

            # Преимущественно вертикальное направление,
            # но с естественным наклоном.
            angle = np.random.normal(0.0, 0.20)

            dx = np.sin(angle) * length
            dy = np.cos(angle) * length

            steps = max(2, int(length * 1.5))

            # Каждая царапина имеет собственную яркость
            # и прозрачность.
            if np.random.rand() < 0.7:
                color = 255.0
            else:
                color = 0.0

            alpha = np.random.uniform(0.20, 0.70)
            alpha *= self.intensity

            thickness = np.random.choice([1, 1, 1, 2])

            for t in np.linspace(0.0, 1.0, steps):
                # Небольшие отклонения создают неровные края.
                jitter_x = np.random.normal(0.0, 0.45)
                jitter_y = np.random.normal(0.0, 0.20)

                x = int(round(x0 + dx * t + jitter_x))
                y = int(round(y0 + dy * t + jitter_y))

                if not (0 <= x < w and 0 <= y < h):
                    continue

                x_min = max(0, x - thickness // 2)
                x_max = min(w, x + (thickness + 1) // 2)
                y_min = max(0, y)
                y_max = min(h, y + 1)

                # Чередование непрозрачных и слабых участков
                # делает царапину менее похожей на прямую линию.
                local_alpha = alpha * np.random.uniform(0.55, 1.0)

                region = alpha_map[y_min:y_max, x_min:x_max]
                stronger = local_alpha > region

                region[stronger] = local_alpha

                color_region = scratches[y_min:y_max, x_min:x_max]
                color_region[stronger] = color

        # Накладываем царапины с разной прозрачностью.
        #mask = alpha_map[:, :, None]
        #result = img * (1.0 - mask) + scratches * mask
        # Слегка смягчаем края царапин.

        soft_alpha = self._blur_mask(alpha_map)
        soft_alpha = np.clip(soft_alpha * 1.15, 0.0, 1.0)

        mask = soft_alpha[:, :, None]
        result = img * (1.0 - mask) + scratches * mask

        # Мелкое плёночное зерно вместо сильного равномерного шума.
        sigma = 4.0 * self.intensity
        noise = np.random.normal(
            0.0, sigma, result.shape
        ).astype(np.float32)

        result += noise

        # Редкие мелкие точки-повреждения.
        speckle_count = int(h * w * 0.00008 * self.intensity)

        if speckle_count > 0:
            ys = np.random.randint(0, h, speckle_count)
            xs = np.random.randint(0, w, speckle_count)

            values = np.random.choice(
                [0.0, 255.0], size=speckle_count
            )

            result[ys, xs] = (
                result[ys, xs] * (1.0 - 0.45 * self.intensity)
                + values[:, None] * (0.45 * self.intensity)
            )

        return np.clip(result, 0, 255).astype(np.uint8)





class Neon(ImageFilter):

    def __init__(self, intensity=1.0, threshold=50, **kwargs):
        self.intensity = max(0.0, min(1.0, intensity))
        self.threshold = max(0, min(255, threshold))
        self.neon_color = np.array([255, 0, 220], dtype=np.float32)

    def apply_filter(self, image):
        if self.intensity == 0:
            return image.copy()

        img = image.astype(np.float32)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        smooth = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0

        gray = self._convolve(gray, smooth)

        gx = self._convolve(gray, np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32))

        gy = self._convolve(gray, np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32))

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        edges = (magnitude > self.threshold).astype(np.float32)

        kernel = np.array([
            [1,  4,  7,  4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1,  4,  7,  4, 1]
        ], dtype=np.float32)
        kernel /= kernel.sum()

        glow_small = edges.copy()
        for _ in range(4):
            glow_small = self._convolve(glow_small, kernel)

        glow_medium = edges.copy()
        for _ in range(14):
            glow_medium = self._convolve(glow_medium, kernel)

        glow_wide = edges.copy()
        for _ in range(30):
            glow_wide = self._convolve(glow_wide, kernel)
        

        glow = (
            glow_small[:, :, None] * np.array([180, 0, 220], dtype=np.float32) * 1.0 +
            glow_medium[:, :, None] * np.array([220, 0, 255], dtype=np.float32) * 2.0 +
            glow_wide[:, :, None] * np.array([255, 0, 180], dtype=np.float32) * 2.5
        )
        

        background = img * 0.10
        result = background + glow

        core = self._convolve(edges, smooth)
        core = core[:, :, None]

        result = result * (1.0 - core * 0.85) + 255.0 * (core * 0.85)

        result = (
            img * (1.0 - self.intensity) +
            result * self.intensity
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _convolve(image, kernel):
        h, w = image.shape
        kh, kw = kernel.shape
        py, px = kh // 2, kw // 2

        padded = np.pad(
            image,
            ((py, py), (px, px)),
            mode="constant"
        )

        result = np.zeros((h, w), dtype=np.float32)

        for y in range(kh):
            for x in range(kw):
                result += (
                    kernel[y, x] *
                    padded[y:y + h, x:x + w]
                )

        return result


