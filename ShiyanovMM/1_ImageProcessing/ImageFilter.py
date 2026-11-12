import cv2
import numpy as np
import matplotlib.pyplot as plt
import random, argparse
import os
from math import sqrt, exp, pi

class ImageFilter:
    @staticmethod
    def get_filter(filter_type, **kwargs):
        filters = {
            "resolution": Resolution_Change,
            "grayscale": Grayscale,
            "antique": Antique,
            "fade": Fade,
            "infrared": Infrared,
            "matte": Matte,
            "noise": Noise,
            "neon": Neon,
        }
        if filter_type not in filters:
            raise ValueError(f"Неизвестный фильтр: {filter_type}")
        return filters[filter_type](**kwargs)

    def apply_filter(self, img):
        pass

class Resolution_Change(ImageFilter):
    def __init__(self, scale=2.0):
        self.scale = scale

    def apply_filter(self, img):
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.scale), int(w * self.scale)
        
        src_i = np.arange(new_h) / self.scale
        src_j = np.arange(new_w) / self.scale

        i0 = np.floor(src_i).astype(int)
        j0 = np.floor(src_j).astype(int)
        i1 = np.minimum(i0 + 1, h - 1)
        j1 = np.minimum(j0 + 1, w - 1)

        i00 = img[i0[:, None], j0[None, :]]
        i01 = img[i0[:, None], j1[None, :]]
        i10 = img[i1[:, None], j0[None, :]]
        i11 = img[i1[:, None], j1[None, :]]

        di = (src_i - i0)[:, None]
        dj = (src_j - j0)[None, :]
        di = di[..., None]
        dj = dj[..., None]

        top = i00 * (1 - dj) + i01 * dj
        bottom = i10 * (1 - dj) + i11 * dj
        result = top * (1 - di) + bottom * di

        return np.clip(result, 0, 255).astype(np.uint8)

class Grayscale(ImageFilter):
    def apply_filter(self, img):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)

class Antique(ImageFilter):
    def __init__(self, intensity=1.0, vignette_color="black"):
        self.intensity = intensity
        self.vignette_color = vignette_color
    def apply_filter(self, img):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        r_new = 0.393 * r + 0.769 * g + 0.189 * b
        g_new = 0.349 * r + 0.686 * g + 0.168 * b
        b_new = 0.272 * r + 0.534 * g + 0.131 * b

        r_new = np.clip(r_new, 0, 255)
        g_new = np.clip(g_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        r_out = r_new * self.intensity + r * (1 - self.intensity)
        g_out = g_new * self.intensity + g * (1 - self.intensity)
        b_out = b_new * self.intensity + b * (1 - self.intensity)

        sepia = np.stack([b_out, g_out, r_out], axis=2)

        h, w = img.shape[:2]

        Y, X = np.ogrid[:h, :w]
        dist_x = np.minimum(X, w - 1 - X)
        dist_y = np.minimum(Y, h - 1 - Y)
        dist = np.minimum(dist_x, dist_y)
        border = min(h, w) * 0.25
        vignette = np.clip(dist/(border+border*self.intensity), 0, 1)

        vignette = 1 - (self.intensity) * (1 - vignette)

        vignette = vignette[:, :, None]

        if self.vignette_color == "white":
            result = sepia * vignette + 222 * (1 - vignette)
        else:
            result = sepia * vignette
        return np.clip(result, 0, 255).astype(np.uint8)

class Fade(ImageFilter):
    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def apply_filter(self, img):
        image = img.astype(np.float32)
        diff = image-128
        result = 128+diff*(1-self.intensity)

        return np.clip(result, 0, 255).astype(np.uint8)

class Infrared(ImageFilter):

    def apply_filter(self, img):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        r_new = 0.1*r+0.9*g
        g_new = 0.3*r+0.7*b
        b_new = 0.1*r+0.1*g+0.8*b

        result = np.stack([b_new, g_new, r_new], axis=2).astype(np.float32)

        return np.clip(result, 0, 255).astype(np.uint8)


class Matte(ImageFilter):
    def apply_filter(self, img):
        h, w = img.shape[:2]
        img = img.astype(np.float64)
        mask = cv2.imread("mask.png")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        result = img+mask
        return np.clip(result, 0, 255).astype(np.uint8)

class Noise(ImageFilter):
    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def apply_filter(self, img):
        result = img.astype(np.float32)
        h, w = img.shape[:2]
        noise_arr = np.random.normal(0, 255 * self.intensity*0.4, result.shape)
        result = result + noise_arr
        for _ in range(round(40*self.intensity)):
            x = np.random.randint(0, w)
            y1 = np.random.randint(0, h // 2)
            y2 = np.random.randint(h // 2, h)
            thickness = np.random.randint(1, 3)
            color = np.random.randint(100, 150)
            result[y1:y2, x:x + thickness] = color
        return np.clip(result, 0, 255).astype(np.uint8)

class Neon(ImageFilter):
    def __init__(self, brightness=0.1):
        self.brightness = brightness

    def apply_filter(self, img):
        h, w = img.shape[:2]
        gaussian_blur = np.zeros((5, 5)).astype(np.float32)
        sigma = 1
        for i in range(-2, 3):
            for j in range(-2, 3):
                d = sqrt(i**2+j**2)
                gaussian_blur[i+2, j+2] = exp(-(d**2)/(2*sigma**2))/sqrt(2*pi*sigma**2)

        
        edge_x =  np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]], dtype=np.float32)

        edge_y = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]], dtype=np.float32)
        
        blur = cv2.filter2D(img, -1, gaussian_blur)
        edge_x = cv2.filter2D(blur, cv2.CV_32F, edge_x)
        edge_y = cv2.filter2D(blur, cv2.CV_32F, edge_y)
        edges = np.hypot(edge_x, edge_y)

        dark = (img.astype(np.float32) * self.brightness * 0.9).astype(np.uint8)
        
        glow_color = edges.astype(np.float32)
        result = dark + glow_color * (1.5-(1-self.brightness)*0.5)
        return np.clip(result, 0, 255).astype(np.uint8)

def unit_float(s):
    v = float(s)
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"{v} не является корректным значением параметра")
    return v
def non_negative_float(s):
    v = float(s)
    if not (0.0 <= v):
        raise argparse.ArgumentTypeError(f"{v} не является корректным значением параметра")
    return v
def bw_str(bw):
    if bw not in ("black", "white"):
        raise argparse.ArgumentTypeError(f"{bw} не является корректным значением параметра")
    return bw
def cli_argument_parser():

    parser = argparse.ArgumentParser(description="Приложение для наложения фильтров на изображения")

    parser.add_argument("input_image", type=str, help="Путь к исходному изображению")
    parser.add_argument("output_image", type=str, help="Имя обработанного изображения")
    parser.add_argument("-f", "--filter", type=str, required=True, 
                        help="Тип применяемого фильтра")
    parser.add_argument("-s", "--scale", type=non_negative_float, default=2.0, 
                        help="Увеличение масштаба (по умолчанию 2)")
    parser.add_argument("-b", "--brightness", type=unit_float, default=0.1, 
                        help="Яркость картинки для фильтра Neon (от 0 до 1, по умолчанию 0.1)")
    parser.add_argument("-i", "--intensity", type=unit_float, default=1.0, 
                        help="Интенсивность фильтра (от 0 до 1, по умолчанию 1)")
    parser.add_argument("-v", "--vignette_color", type=bw_str, default="black", 
                        help="Цвет виньетки (black/white, по умолчанию black)")
    args = parser.parse_args()
    return args

def read_image(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Не удалось прочитать файл: {img_name}. Возможно, он поврежден или это не изображение.")
    return img

def main():
    args = cli_argument_parser()
    img = read_image(args.input_image)
    
    # Подготовка параметров под конкретный фильтр
    kwargs = {}
    if args.filter == "resolution":
        kwargs["scale"] = args.scale
    elif args.filter == "neon":
        kwargs["brightness"] = args.brightness
    elif args.filter in ("fade", "noise"):
        kwargs["intensity"] = args.intensity
    elif args.filter == "antique":
        kwargs["intensity"] = args.intensity
        kwargs["vignette_color"] = args.vignette_color
    # Создание и применение фильтра
    filter_obj = ImageFilter.get_filter(args.filter, **kwargs)
    result = filter_obj.apply_filter(img)
    filename, ext = os.path.splitext(args.output_image)
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"Предупреждение: формат {ext} не поддерживается. Сохраняем как .jpg")
        args.output_image = filename + ".jpg"
    # Сохранение результата
    cv2.imwrite(args.output_image, result)
    print(f"Готово! Результат сохранен успешно в {args.output_image}")

if __name__ == "__main__":
    main()