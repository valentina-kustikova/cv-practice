import cv2
import numpy as np

def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


class ImageFilter:

    def apply_filter(self, image):
        raise NotImplementedError("Метод должен быть переопределен")

    @staticmethod
    def get_filter(filter_name, **kwargs):
        if filter_name == 'resize':
            return ResizeFilter(
                scale=kwargs.get('scale'),
                width=kwargs.get('width'),
                height=kwargs.get('height')
            )
        elif filter_name == 'gray':
            return GrayScale()
        elif filter_name == 'antique':
            return AntiqueFilter(
                vignette=kwargs.get('vignette', 0.6),
                vignette_scale=kwargs.get('vignette_scale', 0.9),
                feather=kwargs.get('feather', 0.3)
            )
        elif filter_name == 'fade':
            return FadeFilter(
                contrast=kwargs.get('contrast', 0.6),
                brightness=kwargs.get('brightness', 60)
            )
        elif filter_name == 'film':
            return FilmFilter()
        elif filter_name == 'matte':
            return MatteFilter(
                margin=kwargs.get('margin', 0.1)
            )
        elif filter_name == 'scratches':
            return ScratchesNoiseFilter(
                noise_level = kwargs.get('noise_level', 20),
                scratches_count = kwargs.get('scratches_count', 15)
            )
        elif filter_name == 'neon':
            return NeonFilter(
                thickness = kwargs.get('thickness', 2),
                blur_size = kwargs.get('blur_size', 15)
            )
        else:
            raise ValueError(f"Неизвестный фильтр: {filter_name}")


class ResizeFilter(ImageFilter):
    def __init__(self, width=None, height=None, scale=None):
        self.width = width
        self.height = height
        self.scale = scale

    def apply_filter(self, image):
        h, w = image.shape[:2]
        if self.scale:
            new_w, new_h = round(w * self.scale), round(h * self.scale)
        elif self.width and self.height:
            new_w, new_h = self.width, self.height
        else:
            raise ValueError('Для resize задайте --scale или --width и --height')

        ys = np.clip((np.arange(new_h) + 0.5) * h / new_h - 0.5, 0, h - 1)
        xs = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5, 0, w - 1)
        y0, x0 = ys.astype(int), xs.astype(int)
        y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
        dy = (ys - y0)[:, None, None]
        dx = (xs - x0)[None, :, None]
        img = image.astype(np.float64)
        top = img[y0][:, x0] * (1 - dx) + img[y0][:, x1] * dx
        bottom = img[y1][:, x0] * (1 - dx) + img[y1][:, x1] * dx
        return to_uint8(np.round(top * (1 - dy) + bottom * dy))


class GrayScale(ImageFilter):
    def apply_filter(self, image):
        b,g,r = image[:, :, 0].astype(np.float64), image[:, :, 1].astype(np.float64), image[:, :, 2].astype(np.float64)
        return to_uint8(0.299 * r + 0.587 * g + 0.114 * b)




class AntiqueFilter(ImageFilter):
    def __init__(self, vignette=0.6, vignette_scale=0.9, feather=0.3):
        if not 0 <= vignette <= 1:
            raise ValueError("Сила виньетки (vignette) должна быть от 0 до 1")
        if not 0 < vignette_scale <= 2:
            raise ValueError("Масштаб виньетки (vignette_scale) должен быть больше 0 и до 2")
        if feather <= 0:
            raise ValueError("Мягкость (feather) должна быть больше 0")

        self.vignette = vignette
        self.vignette_scale = vignette_scale
        self.feather = feather

    def apply_filter(self, image):

        gray_image = GrayScale().apply_filter(image)

        blue_channel = gray_image * 0.43
        green_channel = gray_image * 0.74
        red_channel = gray_image * 1.07

        sepia_image = np.stack([blue_channel, green_channel, red_channel], axis=-1)

        H, W = image.shape[:2]

        cx, cy = W / 2, H / 2
        a = W / 2 * self.vignette_scale
        b = H / 2 * self.vignette_scale

        x_coords = np.arange(W)
        y_coords = np.arange(H)

        dx = (x_coords - cx)[None, :] / a
        dy = (y_coords - cy)[:, None] / b

        d = np.sqrt(dx ** 2 + dy ** 2)

        t = np.clip((1 + self.feather - d) / (2 * self.feather), 0, 1)
        mask = t * t * (3 - 2 * t)
        mask_3ch = mask[..., None]
        result = sepia_image * (1 - self.vignette + self.vignette * mask_3ch)

        return np.clip(result, 0, 255).astype(np.uint8)


class FadeFilter(ImageFilter):
    def __init__(self, contrast=0.6, brightness=60):
        self.contrast = contrast
        self.brightness = brightness

    def apply_filter(self, image):
        img_float = image.astype(np.float64)

        faded = img_float * self.contrast + self.brightness
        return to_uint8(faded)


class FilmFilter(ImageFilter):

    def apply_filter(self, image):
        image = image.astype(np.float64)
        R = image[..., 0] * 1.1 + 10
        G = image[..., 1] * 1.2 + 15
        B = image[..., 2] * 0.7
        return np.clip(np.stack([B, G, R], axis=-1), 0, 255).astype(np.uint8)


class MatteFilter(ImageFilter):
    def __init__(self, margin=0.1):
        if not 0 <= margin <= 0.5:
            raise ValueError("Размер рамки должен быть от 0 до 0.5")
        self.margin = margin

    def apply_filter(self, image):
        h, w = image.shape[:2]

        Y, X = np.ogrid[:h, :w]

        cx, cy = w / 2, h / 2
        rx, ry = w / 2 - w * self.margin, h / 2 - h * self.margin

        distance = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2

        mask = np.clip(10.5 - (10 * distance), 0, 1)
        mask_3ch = mask[..., None]

        white_bg = np.ones_like(image) * 255
        result = image * mask_3ch + white_bg * (1 - mask_3ch)

        return np.clip(result, 0, 255).astype(np.uint8)



class ScratchesNoiseFilter(ImageFilter):

    def __init__(self, noise_level=15, scratches_count=30):
        self.noise_level = noise_level
        self.scratches_count = scratches_count

    def apply_filter(self, image):
        result = AntiqueFilter().apply_filter(image).astype(np.float32)
        h, w = result.shape[:2]

        noise = np.random.normal(0, self.noise_level, (h, w, 1))
        color_noise = np.random.normal(0, self.noise_level * 0.3, image.shape)
        result += noise + color_noise
        result = np.clip(result, 0, 255)

        for _ in range(self.scratches_count):
            x_start = np.random.randint(0, w)
            y_start = np.random.randint(0, h)

            length = np.random.randint(10, max(11, int(h * 0.25)))

            angle = np.random.uniform(-0.45, 0.45)

            thickness = np.random.choice([0, 0, 0, 0, 1])

            t = np.linspace(0, 1, length)
            deviation = np.cumsum(np.random.normal(0, 0.2, length))

            x = x_start + np.sin(angle) * t * length + deviation
            y = y_start + np.cos(angle) * t * length

            keep = np.random.random(length) > 0.15

            brightness = np.random.choice([
                np.random.randint(20, 90),
                np.random.randint(180, 256)
            ])

            alpha = np.random.uniform(0.15, 0.30)

            for offset in range(-thickness, thickness + 1):
                x_pos = np.round(x).astype(int)
                y_pos = np.round(y).astype(int)

                valid = (
                        keep &
                        (x_pos >= 0) & (x_pos < w) &
                        (y_pos >= 0) & (y_pos < h)
                )

                result[y_pos[valid], x_pos[valid]] = (
                        result[y_pos[valid], x_pos[valid]] * (1 - alpha)
                        + brightness * alpha
                )

        return np.clip(result, 0, 255).astype(np.uint8)


class NeonFilter(ImageFilter):
    def __init__(self, thickness=2, blur_size=15):
        self.thickness = thickness
        self.blur_size = blur_size

    def apply_filter(self, image):
        gray = GrayScale().apply_filter(image).astype(np.float64)

        shift = max(1, self.thickness)
        grad_x = np.abs(np.roll(gray, -shift, axis=1) - np.roll(gray, shift, axis=1))
        grad_y = np.abs(np.roll(gray, -shift, axis=0) - np.roll(gray, shift, axis=0))

        edges = np.clip(grad_x + grad_y, 0, 255)

        if self.thickness > 1:
            for _ in range(self.thickness - 1):
                edges = np.maximum(edges, np.roll(edges, 1, axis=0))
                edges = np.maximum(edges, np.roll(edges, -1, axis=0))
                edges = np.maximum(edges, np.roll(edges, 1, axis=1))
                edges = np.maximum(edges, np.roll(edges, -1, axis=1))
        if self.blur_size > 0:
            glow = np.zeros_like(edges)
            for i in range(1, self.blur_size + 1):
                glow += np.roll(edges, i, axis=0)
                glow += np.roll(edges, -i, axis=0)
                glow += np.roll(edges, i, axis=1)
                glow += np.roll(edges, -i, axis=1)
            glow = glow / (4 * self.blur_size)
        else:
            glow = np.zeros_like(edges)

        neon_intensity = np.clip(edges + glow, 0, 255)

        neon_layer = np.zeros_like(image, dtype=np.float64)
        neon_layer[:, :, 0] = neon_intensity
        neon_layer[:, :, 1] = neon_intensity * 0.3
        neon_layer[:, :, 2] = neon_intensity

        dark_bg = image.astype(np.float64) * 0.4
        result = np.maximum(dark_bg, neon_layer)

        return to_uint8(result)
