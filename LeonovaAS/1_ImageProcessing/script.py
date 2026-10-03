from abc import ABC, abstractmethod
import argparse
import numpy as np
import cv2 as cv

class ImageFilter(ABC):
    """Абстрактный класс фильтра"""
    @staticmethod
    def get_filter(filter_name, **kwargs):
        if filter_name == "rescale":
            return Rescale(**kwargs)
        if filter_name == "grayscale":
            return Grayscale(**kwargs)
        if filter_name == "antique":
            return Antique(**kwargs)
        if filter_name == "fade":
            return Fade(**kwargs)
        if filter_name == "ir":
            return IR(**kwargs)
        if filter_name == "matte":
            return Matte(**kwargs)
        if filter_name == "age":
            return Age(**kwargs)
        if filter_name == "neon":
            return Neon(**kwargs)
        
    @abstractmethod
    def apply_filter(self, image):
        pass
    
class Rescale(ImageFilter):
    """Изменение разрешения картинки"""
    def __init__(self, width=100, height=100):
        if width <= 0 or height <= 0:
            raise ValueError(f"Negative dimentions")
        self.w = width
        self.h = height
        
    def apply_filter(self, image):
        h_in, w_in = image.shape[:2]

        scale_x = w_in / self.w
        scale_y = h_in / self.h

        x_out = np.arange(self.w)   # (W_out,)
        y_out = np.arange(self.h)

        x_in = (x_out[None, :] + 0.5) * scale_x - 0.5   # (1, W_out,)
        y_in = (y_out[:, None] + 0.5) * scale_y - 0.5   # (H_out, 1)

        x_idx = np.round(x_in).astype(int)
        y_idx = np.round(y_in).astype(int)

        x_idx = np.clip(x_idx, 0, w_in - 1)
        y_idx = np.clip(y_idx, 0, h_in - 1)

        return image[y_idx, x_idx]

class Grayscale(ImageFilter):
    """Перевод изображения в оттенки серого"""       
    def apply_filter(self, image):
        return np.clip(image[...,0]*0.114
        + image[...,1]*0.587
        + image[...,2]*0.299, 
        0, 255).astype(np.uint8)

class Antique(ImageFilter):
    """Фильтр антиквариат (сепия)"""
    def apply_filter(self, image):
        result = Grayscale().apply_filter(image) 
        B = result * 0.43
        G = result * 0.74
        R = result * 1.07
        result = np.stack([B, G, R], axis=-1)
        return np.clip(result, 0, 255).astype(np.uint8)
    
class Fade(ImageFilter):
    """Выцветание фотографии"""
    def apply_filter(self, image):
        result = Grayscale().apply_filter(image).astype(np.float32)
        B = (result + (image[...,0] - result) * 0.6) * 0.8 + 128 * 0.2
        G = (result + (image[...,1] - result) * 0.6) * 0.8 + 128 * 0.2
        R = (result + (image[...,2] - result) * 0.6) * 0.8 + 128 * 0.2
        return np.clip(np.stack([B, G, R], axis=-1), 0, 255).astype(np.uint8)

class IR(ImageFilter):
    """Эффект инфракрасной пленки"""
    def apply_filter(self, image):
        R = (image[...,0] * 1.1) + 10
        G = (image[...,1] * 1.2) + 15
        B = (image[...,2] * 0.9)
        return np.clip(np.stack([B, G, R], axis=-1), 0, 255).astype(np.uint8)

class Matte(ImageFilter):
    """Эффект белой маски"""
    def __init__(self, scale = 0.8, feather = 0.3):
        if (scale <= 0 or scale > 1):
            raise ValueError(f"Invalid scale value")
        if feather <= 0:
            raise ValueError(f"Negative feather value")
        self.scale = scale
        self.feather = feather
        
    def apply_filter(self, image):
        H, W = image.shape[:2]
        
        a, b = W * self.scale / 2, H * self.scale / 2
        cx, cy = W / 2, H / 2
        
        x = np.arange(W)
        y = np.arange(H)
        
        dx = (x - cx)[None, :] / a # (1, W)
        dy = (y - cy)[:, None] / b # (H, 1)

        d = np.sqrt(dx**2 + dy**2) # (H, W)
        
        mask = (1 + self.feather - d) / (2 * self.feather)
        mask = np.clip(mask, 0, 1)
        mask_3ch = mask[..., None] # (H, W, 1)
        
        result = image * mask_3ch + 255 * (1 - mask_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

class Age(ImageFilter):
    """Эффект состаривания картинки"""
    def __init__(self, sigma = 10.0, seed = None):
        if sigma < 0:
            raise ValueError("Negative sigma value")
        self.sigma = sigma
        self.seed = seed
        
    def apply_filter(self, image):
        rng = np.random.default_rng(self.seed)
        
        H,W = image.shape[:2]
        
        result = image.astype(np.float32)
        
        if self.sigma > 0:
            noise = rng.normal(0, self.sigma, (H, W, 1))
            result = result + noise
            
        scratches = int(rng.integers(5, 26))
        scratch_mask = np.zeros((H, W), dtype = np.float32)
        
        diag = int(np.sqrt(H**2 + W**2))

        for _ in range(scratches):
            x0 = int(rng.integers(0, W))
            y0 = int(rng.integers(0, H))

            length = int(rng.integers(diag // 20, diag // 3 + 1)) #от 5 до 30% диагонали картинки
            angle = rng.uniform(0, 2 * np.pi)
            
            if rng.random() < 0.7:
                intensity = float(rng.uniform(30, 80)) #светлая
            else:
                intensity = float(rng.uniform(-80, -30)) #тёмная

            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 + length * np.sin(angle))

            cv.line(scratch_mask, (x0, y0), (x1, y1), color=intensity, thickness=1)
        
        result = result + scratch_mask[..., None]
        return np.clip(result, 0, 255).astype(np.uint8)
        
class Neon(ImageFilter):
    """Неоновое подсвечивание"""
    def __init__(self, color = "pink", intensity = 1.5):
        if (color != "pink") and (color != "cyan") and (color != "green"):
            raise ValueError("Color can only be pink, cyan or green")
        if intensity < 0:
            raise ValueError("Negative intensity value")
        if color == "pink":
            self.color = np.array((149,71,255), dtype=np.float32)
        if color == "cyan":
            self.color = np.array((238,255,71), dtype=np.float32)
        if color == "green":
            self.color = np.array((17,255,0), dtype=np.float32)
        self.intensity = intensity
        
    def apply_filter(self, image):
        gray = (Grayscale().apply_filter(image)).astype(np.float32)
        
        Gx = np.zeros_like(gray)
        Gy = np.zeros_like(gray)
        Gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        Gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        G = np.sqrt(Gx**2 + Gy**2)
        
        if G.max() > 0:
            G = G / G.max()
        G = G ** 0.7
        
        padded = np.pad(G, 1, mode='edge')
        glow = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
            padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        ) / 9.0
        
        neon = (G[..., None] + glow[..., None] * 0.5) * self.color * self.intensity
        
        result = image.astype(np.float32) * 0.2 + neon

        return np.clip(result, 0, 255).astype(np.uint8)                

def parse_args(args):
    """Отдельный разбор строки параметров типа width=124,height=124"""
    result = {}
    if args != "":
        for pair in args.split(","):
            key, value = pair.split("=")
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass               
            result[key.strip()] = value         
    return result

def cli_argument_parser():
    """Разбор аргументов из командной строки --filter --filter_args --image"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--filter", required=True, 
    choices=["rescale", "grayscale", "antique", "fade", "ir", "matte", "age", "neon"])
    
    parser.add_argument("--filter_args", default="")
    
    parser.add_argument("--image", required=True)
    
    args = parser.parse_args()
    return args
    
def read_image(image_name):
    """Чтение картинки по пути"""
    image = cv.imread(image_name, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found: {image_name}")
    return image
    
def main():
    args = cli_argument_parser()
    kwargs = parse_args(args.filter_args)
    image = read_image(args.image)
    
    f1 = ImageFilter.get_filter(args.filter, **kwargs)
    result = f1.apply_filter(image)
    result_name = f"{args.filter}_{args.image}"
    cv.imwrite(result_name, result)

if __name__ == "__main__":
    main()
    