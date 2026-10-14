import numpy as np
from .base import ImageFilter


class FadeColor(ImageFilter): # Выцветание
    def __init__(self, strength=0.5):
        if not 0 <= strength <= 1:
            raise ValueError('strength должен быть в [0, 1]')
        self.strength = strength

    def apply_filter(self, image):
        # белая картинка того же размера, что исходная
        white = np.full_like(image, 255, dtype=np.float32) # full_like создаёт массив формы, как image, заполненный значением 255
        img = image.astype(np.float32) # сохраняем дробные числа чтобы не терять точность

        # I' = I*(1-s) + 255*s - формула выцветания: I — исходный пикс. 255 — белый. s — сила выцветания 
        # линейно смешиваем оригинал с белым: линейная интерполяция между исходным пикселем I и белым 255 — это взвешенное среднее между двумя значениями.
        result = img * (1 - self.strength) + white * self.strength
        
        return np.clip(result, 0, 255).astype(np.uint8) # обрезает значения до допустимых и превращает в целые числа