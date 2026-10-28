import numpy as np
import cv2 as cv

from filter_base import ImageFilter
   
class Rescale(ImageFilter):
    """Изменение разрешения картинки"""
    def __init__(self, width=100, height=100):
        if width <= 0 or height <= 0:
            raise ValueError(f"Negative dimensions")
        self.w = width
        self.h = height
        
    def apply_filter(self, image):
        h_in, w_in = image.shape[:2]

        scale_x = w_in / self.w
        scale_y = h_in / self.h

        x_out = np.arange(self.w) #(W_out,)
        y_out = np.arange(self.h)

        x_in = (x_out[None, :] + 0.5) * scale_x - 0.5 #(1, W_out,)
        y_in = (y_out[:, None] + 0.5) * scale_y - 0.5 #(H_out, 1)

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
    def __init__(self, vignette=0.6, vignette_scale=0.9, feather=0.3):
        if not 0 <= vignette <= 1:
            raise ValueError("vignette must be in [0, 1]")
        if not 0 < vignette_scale <= 2:
            raise ValueError("vignette_scale must be in (0, 2]")
        if feather <= 0:
            raise ValueError("Negative feather")
        self.vignette = vignette
        self.vignette_scale = vignette_scale
        self.feather = feather
    
    def apply_filter(self, image):
        sepia = Grayscale().apply_filter(image) 
        B = sepia * 0.43
        G = sepia * 0.74
        R = sepia * 1.07
        sepia = np.stack([B, G, R], axis=-1)
        
        H, W = image.shape[:2]
        
        cx, cy = W / 2, H / 2
        a = W / 2 * self.vignette_scale
        b = H / 2 * self.vignette_scale

        dx = (np.arange(W) - cx)[None, :] / a
        dy = (np.arange(H) - cy)[:, None] / b
        d = np.sqrt(dx**2 + dy**2)

        t = np.clip((1 + self.feather - d) / (2 * self.feather), 0, 1)
        mask = t * t * (3 - 2 * t)

        mask_3ch = mask[..., None]
        result = sepia * (1 - self.vignette + self.vignette * mask_3ch)

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
        
        dx = (x - cx)[None, :] / a #(1, W)
        dy = (y - cy)[:, None] / b #(H, 1)

        d = np.sqrt(dx**2 + dy**2) #(H, W)
        
        mask = (1 + self.feather - d) / (2 * self.feather)
        mask = np.clip(mask, 0, 1)
        mask_3ch = mask[..., None] #(H, W, 1)
        
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
                intensity = float(rng.uniform(10, 20)) #светлая
            else:
                intensity = float(rng.uniform(-25, -10)) #тёмная

            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 + length * np.sin(angle))

            cv.line(scratch_mask, (x0, y0), (x1, y1), color=intensity, thickness=1)
        
        result = result + scratch_mask[..., None]
        return np.clip(result, 0, 255).astype(np.uint8)
        
class Neon(ImageFilter):
    """Неоновое подсвечивание как в фотошопе"""
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
        
    @staticmethod
    def _gaussian_kernel(k, sigma):
        """Гауссово ядро размера"""
        r = k // 2
        x = np.arange(-r, r + 1, dtype=np.float32)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    @classmethod
    def _gaussian_blur(cls, arr, k, sigma):
        """Гауссово размытие"""
        kernel = cls._gaussian_kernel(k, sigma)
        r = k // 2
        H, W = arr.shape

        padded_h = np.pad(arr, ((0, 0), (r, r)), mode='edge')
        tmp = np.zeros_like(arr, dtype=np.float32)
        for i in range(k):
            tmp += padded_h[:, i:i + W] * kernel[i]

        padded_v = np.pad(tmp, ((r, r), (0, 0)), mode='edge')
        result = np.zeros_like(arr, dtype=np.float32)
        for i in range(k):
            result += padded_v[i:i + H, :] * kernel[i]

        return result
        
    def apply_filter(self, image):
        gray = (Grayscale().apply_filter(image)).astype(np.float32)
        
        Gx = np.zeros_like(gray)
        Gy = np.zeros_like(gray)
        Gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        Gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        G = np.sqrt(Gx ** 2 + Gy ** 2)

        if G.max() > 0:
            G = G / G.max()
        G = G ** 0.9

        glow_narrow = self._gaussian_blur(G, k=5,  sigma=2.5)
        glow_medium = self._gaussian_blur(G, k=15, sigma=4.0)
        glow_wide   = self._gaussian_blur(G, k=41, sigma=11.0)

        WHITE = np.array((255, 255, 255), dtype=np.float32)

        core   = G[..., None] * WHITE * 1.2
        narrow = glow_narrow[..., None] * self.color * 0.8
        medium = glow_medium[..., None] * self.color * 0.4
        wide   = glow_wide[..., None] * self.color * 0.35

        neon = core + narrow + medium + wide

        result = image.astype(np.float32) + neon * self.intensity

        return np.clip(result, 0, 255).astype(np.uint8)                