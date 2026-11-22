import numpy as np
from .base import ImageFilter


class Film(ImageFilter):
#имитация плёнки

    def __init__(self, strength: float = 0.35, noise: float = 8.0):
        self.strength = strength #сила эффекта перестановки каналов
        self.noise = noise #шум

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=2)
        
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        #перестановка каналов
        new_r = (1 - self.strength) * r + self.strength * b
        new_g = g * (1 - 0.15 * self.strength)
        new_b = (1 - self.strength) * b + self.strength * r
        #сборка в BGR
        out = np.stack([new_b, new_g, new_r], axis=2)
        #тонирование (синий +5%) и обрезка
        out[..., 0] = np.clip(out[..., 0] * 1.05, 0, 255)
        out[..., 2] = np.clip(out[..., 2] * 0.95, 0, 255) #красный -5

        if self.noise > 0:
            out += np.random.normal(0, self.noise, out.shape)

        return np.clip(out, 0, 255).astype(np.uint8)
    
