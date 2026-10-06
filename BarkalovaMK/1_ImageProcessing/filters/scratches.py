import numpy as np
from .base import ImageFilter


class ScratchesNoise(ImageFilter):
#Состаривание: царапины + шум

    def __init__(self, n_scratches: int = 25, noise_std: float = 10.0,
                 sp_amount: float = 0.02):
        self.n_scratches = n_scratches
        self.noise_std = noise_std
        self.sp_amount = sp_amount

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.ndim == 2: #1канал в 3х
            img = np.stack([img] * 3, axis=2) #три копии в 3ю ось

        h, w = img.shape[:2]

        for _ in range(self.n_scratches): #царапины
            x = np.random.randint(0, w)
            y0 = np.random.randint(0, h)
            length = np.random.randint(h // 8, h)
            y1 = min(h - 1, y0 + length)
            thickness = np.random.randint(1, 3) #ширина
            color = 255 if np.random.rand() > 0.5 else 0
            img[y0:y1, max(0, x - thickness):x + thickness] = color

        if self.noise_std > 0: #шум
            img += np.random.normal(0, self.noise_std, img.shape)

        if self.sp_amount > 0: #импульсивный шум
            n = int(self.sp_amount * h * w) #кол-во пикселей
            #случайные коорд
            ys = np.random.randint(0, h, n)
            xs = np.random.randint(0, w, n)
            vals = np.where(np.random.rand(n) > 0.5, 255, 0) #случ знач
            img[ys, xs] = vals[:, None]

        return np.clip(img, 0, 255).astype(np.uint8)