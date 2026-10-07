import numpy as np
from .base import ImageFilter


class Neon(ImageFilter):
#Неоновый эффект: выделение контуров оператором Собеля

    def __init__(self, threshold: int = 40, glow: float = 1.2):
        self.threshold = threshold   #порог для опр-я контуров
        self.glow = glow  #множитель яркости градиента

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.ndim == 3:
            gray = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1]
                    + 0.114 * img[:, :, 0])
        else:
            gray = img

        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = kx.T

        def conv2d(a, k): #свёртка изображения a с ядром k
            ph, pw = a.shape 
            out = np.zeros_like(a, dtype=np.float32) #массив 0 того же размера
            for i in range(1, ph - 1):
                for j in range(1, pw - 1):
                    out[i, j] = np.sum(a[i - 1:i + 2, j - 1:j + 2] * k)
            return out

        gx = conv2d(gray, kx) #горизонт-ый градиент
        gy = conv2d(gray, ky) #вертикальный
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mag = np.clip(mag * self.glow, 0, 255)
        #массив булевых значений (H, W) 1 на границах, 0 в однородных областях
        edges = (mag > self.threshold).astype(np.float32) #порог

        if img.ndim == 3:
            base = img * 0.25 #ориг приглушается в 4 раза
            color_glow = np.stack([
                edges * 0.2 * mag,
                edges * 0.9 * mag,
                edges * 1.0 * mag,
            ], axis=2)
            out = base + color_glow
        else:
            out = edges * mag

        return np.clip(out, 0, 255).astype(np.uint8)