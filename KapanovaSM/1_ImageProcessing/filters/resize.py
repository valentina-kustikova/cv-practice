# Изменение разрешения 

import numpy as np
from .base import ImageFilter


class Resize(ImageFilter):

    def __init__(self, scale=0.5):
        if scale <= 0:
            raise ValueError('scale должен быть > 0')
        self.scale = scale

    def apply_filter(self, image):           # Метод ближайшего соседа через linspace
        h, w = image.shape[:2]               # размеры исходной картинки.
        
        # Считаем новые размеры, умножая старые на scale, и не даём им стать нулём
        new_h = max(int(h * self.scale), 1)  # self.scale — параметр кот .пользователь передал
        new_w = max(int(w * self.scale), 1)  # if scale мал то int=0. max(0, 1) = 1 гарантируем хоть 1 пиксель


        # linspace + округление вниз
        # Создаём список номеров строк и столбцов, равномерно распределённых по всей картинке, и округляем их вниз до целых
        # linspace(0, h - 1, new_h) делает new_h равномерно распр чисел от 0 до h-1 (вкл концы)
        # .astype(int) превращает массив из дробных чисел в целые тк numpy требует int для индексации, а floor возвращает float

        row_idx = np.floor(np.linspace(0, h - 1, new_h)).astype(int)  # список номеров строк, которые надо взять из исходной картинки
        col_idx = np.floor(np.linspace(0, w - 1, new_w)).astype(int)

        return image[row_idx][:, col_idx] # Берём из исх. картинки только пиксели по этим номерам строк и столбцов — получаем уменьшенную картинку.