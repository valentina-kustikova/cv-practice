import numpy as np
from .base import ImageFilter


class FadeColor(ImageFilter):
# Выцветание: снижение контраста и насыщенности

    def __init__(self, alpha: float = 0.7, brightness: int = 25):
        self.alpha = alpha #крэфф выцвет
        self.brightness = brightness #осветлить после выцвет

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        img = self.alpha * img + (1 - self.alpha) * 255
        img += self.brightness
        #снижение насыщенности
        if img.ndim == 3: #усредняем по третьей оси(по каналам BGR)
            gray = img.mean(axis=2, keepdims=True)
            img = 0.6 * img + 0.4 * gray #смешивание исх с серым

        return np.clip(img, 0, 255).astype(np.uint8)