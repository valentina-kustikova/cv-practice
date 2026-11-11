import numpy as np
from .base import ImageFilter

class Matte(ImageFilter): # Овальная рамка с мягким краем

    def __init__(self, size=0.9, softness=0.15):
        if not 0 < size <= 1:
            raise ValueError('size должен быть в (0, 1]')
        if softness < 0:
            raise ValueError('softness должен быть >= 0')
        self.size = size
        self.softness = softness

    def apply_filter(self, image):
        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0 # центр картинки

        # радиусы эллипса
        a = (w / 2.0) * self.size
        b = (h / 2.0) * self.size

        # Координатная сетка
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) # создаёт две матрицы формы (h, w)
        # Для каждого пикселя dist — нормированное расстояние до центра эллипса.
        dist = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2) # матрица расстояний от каждого пикселя до центра по формуле эллипса.

        # Маска alpha
        if self.softness <= 0: # чёткий край
        # для каждого пикс проверяем: dist <= 1? Возвращает матрицу булевых значений (True/False)
            alpha = (dist <= 1.0).astype(np.float32) # Превращает True/False в 1.0/0.0
        else: # мягкий край
        # Сигмоида — функция плавного перехода от 0 к 1. α = 1 / (1 + exp(x))
            alpha = 1.0 / (1.0 + np.exp((dist - 1.0) / self.softness))
        # альфа - матрица-маска
        alpha = alpha[..., None] # добавляет третью ось размером 1. Форма: было (h, w), стало (h, w, 1). чтобы потом умножить на img формы (h, w, 3)

        img = image.astype(np.float32) # превращает картинку из целых чисел в дробные. Зачем: при умножении на alpha (дробное число) результат будет дробным — нужно место для хранения.
        
        # белая картинка того же размера, что исходная
        white = np.full_like(img, 255, dtype=np.float32) # full_like создаёт массив формы, как image, заполненный значением 255
        result = img * alpha + white * (1 - alpha) # Смешивание с белым. Какой вклад каждой картинки — задаёт alpha
        # alpha — cколько оставить оригинала, (1 - alpha) — вес белого
        # линейно смешиваем оригинал с белым: линейная интерполяция между исходным пикселем I и белым 255 — это взвешенное среднее между двумя значениями.

        return np.clip(result, 0, 255).astype(np.uint8) # обрезает значения до допустимых и превращает в целые числа