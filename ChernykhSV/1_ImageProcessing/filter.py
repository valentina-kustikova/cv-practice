import numpy as np

from abc import ABC, abstractmethod



class ImageFilter(ABC):
    def __init__(self, **params):
        self.params = params


    @abstractmethod
    def apply_filter(self, image):
        ...

    @staticmethod
    def get_filter(filter, **params):
        filters = {
            "resize": Resize,
            "grayscale": RGB2GrayScale,
            "antique": Antique,
            "fade": FadeColor,
            "infrared": Infrared,
            "matte": Matte,
            "aged": AgedPhoto,
            "neon": NeonEffect,
        }
        if filter not in filters:
            raise ValueError(f"Неизвестный фильтр: {filter}")
        return filters[filter](**params)



class Resize(ImageFilter):
    def apply_filter(self, image):
        h, w = image.shape[:2]
        scale = float(self.params.get("scale", 0.5))

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # координаты новых пикселей 
        x = np.linspace(0, w - 1, new_w)
        y = np.linspace(0, h - 1, new_h)

        # 2 матрицы для определения весов исходного изображения
        x0 = np.floor(x).astype(int) # индексы пикселей по оси x
        y0 = np.floor(y).astype(int) # индексы пикселей по оси y 

        # веса изображения на сколько точка сдвинута от левого соседа к правому (от нижнего к верхнему), число из множества [0,1)
        wx = (x - x0)[None, :, None] # [new_w] -> [1, new_w, 1]
        wy = (y - y0)[:, None, None] # [new_h] -> [new_h, 1 , 1]

        x1 = np.clip(x0 + 1, 0, w - 1) # индексы пикселей справа от x0
        y1 = np.clip(y0 + 1, 0, h - 1) # индексы пикселей сверху от y0

        # квадрат внутри которого находится новый пиксель
        a = image[y0][:, x0]  # левый верхний сосед
        b = image[y0][:, x1]  # правый верхний сосед
        c = image[y1][:, x0]  # левый нижний сосед
        d = image[y1][:, x1]  # правый нижний сосед

        top = a * (1 - wx) + b * wx
        bot = c * (1 - wx) + d * wx
        out = top * (1 - wy) + bot * wy

        return np.clip(out, 0, 255).astype(np.uint8)



class RGB2GrayScale(ImageFilter):


    def apply_filter(self, image):
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = np.clip(gray, 0, 255).astype(np.uint8) # обрезаем + откидываем дробную часть
        return np.stack([gray, gray, gray], axis=-1) # все три канала равны, значит пиксель серый


class Antique(ImageFilter):
    
    def apply_filter(self, image):
        sepia = float(self.params.get("sepia", 0.7))
        sepia = max(0.0, min(1.0, sepia))   # ограничиваем 0..1

        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        
        sepia_r = 0.393 * r + 0.769 * g + 0.189 * b
        sepia_g = 0.349 * r + 0.686 * g + 0.168 * b
        sepia_b = 0.272 * r + 0.534 * g + 0.131 * b

        sepia_bgr = np.stack([sepia_b, sepia_g, sepia_r], axis=-1) # [sepia_b, sepia_g, sepia_r]

        original = image.astype(np.float32)
        out = original * (1.0 - sepia) + sepia_bgr * sepia
        return np.clip(out, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def apply_filter(self, image):

        alpha = float(self.params.get("alpha", 0.6))
        level = float(self.params.get("fade-level", 150))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha должен быть в диапазоне [0, 1], получено: {alpha}")
        if not 0.0 <= level <= 255.0:
            raise ValueError(f"fade-level должен быть в диапазоне [0, 255], получено: {level}")

        img = image.astype(np.float32)
        res = img * alpha + level * (1.0 - alpha)

        return np.clip(res, 0, 255).astype(np.uint8)


class Infrared(ImageFilter):
    def apply_filter(self, image):
        infrared_strength = float(self.params.get("infrared_strength", 0.7))
        infrared_strength = max(0.0, min(1.0, infrared_strength))

        img = image.astype(np.float32)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        ir = 0.4 * r + 0.9 * g + 0.1 * b

        new_r = ir
        new_g = 0.5 * b + 0.5 * ir
        new_b = 0.5 * r + 0.5 * ir

        film = np.stack([new_b, new_g, new_r], axis=-1)

        out = img * (1.0 - infrared_strength) + film * infrared_strength
        return np.clip(out, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def apply_filter(self, image):
        radius = float(self.params.get("matte_radius", 0.8))
        softness = float(self.params.get("matte_softness", 0.2))

        if not 0.0 < radius <= 1.0:
            raise ValueError(f"matte_radius должен быть в (0, 1], получено: {radius}")
        if softness < 0:
            raise ValueError(f"matte_softness должен быть ≥ 0, получено: {softness}")

        # grayscale → добавляем каналы
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]

        # квадрат нормализованного расстояния до центра
        d = ((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2

        # weight = 1 снаружи (белый), 0 внутри (изображение)
        if softness == 0:
            weight = (d > radius).astype(np.float32)
        else:
            weight = np.clip((d - radius) / softness, 0, 1).astype(np.float32)

        weight = weight[:, :, None]   # (h, w) -> (h, w, 1) для broadcasting

        out = image.astype(np.float32) * (1.0 - weight) + 255.0 * weight
        return np.clip(out, 0, 255).astype(np.uint8)


class AgedPhoto(ImageFilter):
    def apply_filter(self, image):
        noise = float(self.params.get("noise_amount", 0.05))
        scratches = int(self.params.get("scratches_count", 7))

        if not 0.0 <= noise <= 0.5:
            raise ValueError(f"noise должен быть в [0, 0.5], получено: {noise}")
        if not 0 <= scratches <= 50:
            raise ValueError(f"scratches должен быть в [0, 50], получено: {scratches}")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        res = image.astype(np.float32)
        h, w = res.shape[:2]

        if noise > 0:
            res += np.random.normal(0, noise * 255, (h, w, 1)).astype(np.float32)

        for _ in range(scratches):
            x_start = np.random.randint(0, w)
            y_start = np.random.randint(0, h // 2)
            length = np.random.randint(h // 4, h)
            slope = np.random.uniform(-0.03, 0.03)
            for dy in range(length):
                y = y_start + dy
                if y >= h:
                    break
                xx = int(x_start + slope * dy)
                if 0 <= xx < w:
                    # смешивание с белым на 15%
                    res[y, xx] = res[y, xx] * 0.85 + 255 * 0.15

        return np.clip(res, 0, 255).astype(np.uint8)

class NeonEffect(ImageFilter):
    def apply_filter(self, image):
        gain = float(self.params.get("neon_gain", 1.0))
        if gain <= 0:
            raise ValueError(f"neon_gain должен быть > 0, получено: {gain}")

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        img = image.astype(np.float32)

        # Делаем изображение серым, чтобы найти границы яркости.
        gray = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]

        # Ядра Собеля, чтобы определить места резкого изменения яркости.
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # Добавляем шаблон на каждый пиксель
        h, w = gray.shape
        pad = np.pad(gray, 1, mode="edge")
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        for dy in range(3):
            for dx in range(3):
                gx += Kx[dy, dx] * pad[dy:dy+h, dx:dx+w]
                gy += Ky[dy, dx] * pad[dy:dy+h, dx:dx+w]

        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip(magnitude * gain, 0, 255)

        out = img.copy()
        out[..., 0] += magnitude   # B
        out[..., 1] += magnitude   # G

        return np.clip(out, 0, 255).astype(np.uint8)