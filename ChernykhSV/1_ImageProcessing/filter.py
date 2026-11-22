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

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        sepia_r = 0.393 * r + 0.769 * g + 0.189 * b
        sepia_g = 0.349 * r + 0.686 * g + 0.168 * b
        sepia_b = 0.272 * r + 0.534 * g + 0.131 * b

        sepia_bgr = np.stack([sepia_b, sepia_g, sepia_r], axis=-1)  # [B, G, R]

        original = image.astype(np.float32)
        out = original * (1.0 - sepia) + sepia_bgr * sepia

        # виньетка: затемнение по краям изображения
        h, w = out.shape[:2]
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0 # центр изображения
        max_dist = np.sqrt(cx ** 2 + cy ** 2) # для нормировки дистанции 0..1
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist  # 0 в центре, 1 в углу

        mask = 1.0 - dist ** 2
        out *= mask[..., None]

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

        res = image.astype(np.float32)
        h, w = res.shape[:2]

        if noise > 0: # добавляем шум - нормально распределение: среднее 0, ширина колокола, форма выходного массива
            res += np.random.normal(0, noise * 255, (h, w, 1)).astype(np.float32)

        for _ in range(scratches):
            # разнообразный наклон дубинок
            angle = np.clip(np.random.normal(0.0, np.pi / 6), -np.pi / 3, np.pi / 3)

            length = np.random.randint(h // 4, h)
            x0 = np.random.uniform(-0.1 * w, 1.1 * w)
            y0 = np.random.uniform(-0.1 * h, h // 2)

            t = np.arange(length, dtype=np.float32) # пикселей в царапине
            # плавное искривление вдоль царапины
            drift = np.cumsum(np.random.normal(0.0, 0.002, length)).astype(np.float32) # делаем царапину волнистой

            xs = (x0 + t * np.sin(angle) + drift * np.cos(angle)).astype(np.intp) # идем вдоль угла angle и добавляем перпендикулярный сдвиг drift
            ys = (y0 + t * np.cos(angle) - drift * np.sin(angle)).astype(np.intp)

            inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w) # если точки не лежат внутри картинки, то отсекаем их
            ys, xs, t_in = ys[inside], xs[inside], t[inside]

            if len(t_in) == 0: # если царапина за кадром, то пропускаем итерацию
                continue

            # делаем царапину не сплошной
            keep = np.random.random(len(t_in)) > 0.15
            ys, xs, t_in = ys[keep], xs[keep], t_in[keep]

            if len(t_in) == 0: # если после прореживания ничего не осталось
                continue

            
            freq = np.random.uniform(0.02, 0.08) # равномерное распределение - частота пульсации
            phase = np.random.uniform(0.0, 2 * np.pi) # фаза пульсации
            alpha = 0.30 * (0.6 + 0.4 * np.sin(t_in * freq + phase)) # интенсивность пикселя t_in на царапине

            res[ys, xs] = res[ys, xs] * (1 - alpha[:, None]) + 255 * alpha[:, None]

        return np.clip(res, 0, 255).astype(np.uint8)

class NeonEffect(ImageFilter):
    def apply_filter(self, image):

        neon_gain = float(self.params.get("neon_gain", 40.0)) # насколько выражена граница? чем больше число, тем более выраженная она должна быть в исходном изображении
        neon_brightness = float(self.params.get("neon_brightness", 0.8)) # яркость свечения размытия неона
        neon_brightness_width = int(self.params.get("neon_brightness_width", 10)) # насколько далеко цвет расходится от контура
        edge_thickness = int(self.params.get("edge_thickness", 1)) # сколько раз применить дилатацию

        color_param = self.params.get("color", (255, 0, 139)) # фиолетовый неон!
        cb, cg, cr = map(float, color_param)

        # проверка значений параметров
        if neon_gain <= 0:
            raise ValueError(f"neon_gain должен быть > 0, получено: {neon_gain}")
        if neon_brightness < 0:
            raise ValueError(f"neon_brightness должен быть >= 0, получено: {neon_brightness}")
        if neon_brightness_width < 1:
            raise ValueError(f"neon_brightness_width должен быть >= 1, получено: {neon_brightness_width}")
        if edge_thickness < 0:
            raise ValueError(f"edge_thickness должен быть >= 0, получено: {edge_thickness}")

        img = image.astype(np.float32)
        h, w = img.shape[:2]
        # делаем изображение серым
        gray_img = RGB2GrayScale().apply_filter(img)
        gray = gray_img[..., 0].astype(np.float32) # берем 0-вой канал, т.к. они все одинаковые

        # ядра оператора Собеля
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        # расширяем картинку для применения оператора собеля
        pad = np.pad(gray, 1, mode="edge")
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        for dy in range(3):
            for dx in range(3):
                gx += Kx[dy, dx] * pad[dy:dy+h, dx:dx+w]
                gy += Ky[dy, dx] * pad[dy:dy+h, dx:dx+w]
        
        magnitude = np.sqrt(gx * gx + gy * gy) # степень изменения яркости
        
        # mask - те точки, которые станут неоновыми
        mask = (magnitude > neon_gain).astype(np.float32)

        # дилатация - значение элемента меняется на максимум из окрестности
        for _ in range(edge_thickness):
            out_dilated = mask.copy()
            out_dilated[1:, :] = np.maximum(out_dilated[1:, :], mask[:-1, :])
            out_dilated[:-1, :] = np.maximum(out_dilated[:-1, :], mask[1:, :])
            out_dilated[:, 1:] = np.maximum(out_dilated[:, 1:], mask[:, :-1])
            out_dilated[:, :-1] = np.maximum(out_dilated[:, :-1], mask[:, 1:])
            mask = out_dilated


        # размытие неона по контурам
        sigma = max(1.0, float(neon_brightness_width)) # насколько далеко цвет расходится от контура
        neon_brightness_mask = mask * 0.4 # сделали маску более тусклой, чтобы потом засветить в цикле
        current = mask
        radius = max(1, int(sigma * 3))

        for r in range(1, radius + 1): # еще radius дилатаций
            current_dilated = current.copy()
            current_dilated[1:, :] = np.maximum(current_dilated[1:, :], current[:-1, :])
            current_dilated[:-1, :] = np.maximum(current_dilated[:-1, :], current[1:, :])
            current_dilated[:, 1:] = np.maximum(current_dilated[:, 1:], current[:, :-1])
            current_dilated[:, :-1] = np.maximum(current_dilated[:, :-1], current[:, 1:])
            current = current_dilated

            weight = np.exp(-(r * r) / (2 * sigma * sigma)) * 0.4 # чем больше r, тем меньше коэффициент weight
            neon_brightness_mask = np.maximum(neon_brightness_mask, current * weight)


        edges = np.zeros_like(img)
        edges[..., 0] = mask * cb
        edges[..., 1] = mask * cg
        edges[..., 2] = mask * cr

        neon_layer = np.zeros_like(img)
        neon_layer[..., 0] = neon_brightness_mask * cb
        neon_layer[..., 1] = neon_brightness_mask * cg
        neon_layer[..., 2] = neon_brightness_mask * cr

        out = img + edges + neon_layer * neon_brightness

        return np.clip(out, 0, 255).astype(np.uint8)