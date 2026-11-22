import numpy as np
from .base import ImageFilter


class Antique(ImageFilter): # Антиквариат: Сепия + виньетка + шум

    # Матрица сепии - коричневато-жёлтый оттенок
    SEPIA_MATRIX = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ], dtype=np.float32)

    def __init__(self, vignette_strength: float = 0.4,
                 noise: float = 6.0):
        if vignette_strength < 0:
            raise ValueError('vignette_strength должен быть >= 0')
        if noise < 0:
            raise ValueError('noise должен быть >= 0')
        self.vignette_strength = vignette_strength
        self.noise = noise

    def apply_filter(self, image):
        img = image.astype(np.float32)
        rgb = img[:, :, ::-1]                    # BGR -> RGB
        result_rgb = rgb @ self.SEPIA_MATRIX.T   # каждый пиксель умножается на матрицу сепии 
        result_bgr = result_rgb[:, :, ::-1]      # RGB -> BGR

        # 2. Виньетка (затемнение краёв)
        if self.vignette_strength > 0:
            h, w = image.shape[:2]
            cy, cx = h / 2.0, w / 2.0 # Центр картинки

            # Координатная сетка. cоздаем две матрицы: yy — в каждой ячейке номер строки (Y), xx — в каждой ячейке номер столбца (X)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) # Расстояние от центра. делим на cx и cy — нормализация. Так: В центре d = 0, на краю d = 1
            d = d / d.max() #делим на максимум, чтобы d был в диапазоне 0..1, в т.ч. в углах 
            # это матрица расстояний каждого пикс до центра картинки.

            # v = 1 - s * d² формула виньетки, s — сила виньетки, d — расстояние от центра
            vignette = 1.0 - self.vignette_strength * (d ** 2) # Угол  d=1.0 => 0.4	× 0.4 — сильно темнее, ближе к центру светлее
            vignette = np.clip(vignette, 0.0, 1.0)[..., None] 
            # clip обрезает значения виньетки до диапазона [0, 1], чтобы при умножении пиксели не выходили за допустимые границы.
            # [..., None] добавляет новую ось размером 1 — форма становится (h, w, 1) вместо (h, w), чтоб numpy мог умножить виньетку на все 3 цветовых канала 

            result_bgr = result_bgr * vignette # Умножаем каждый пиксель на его v => края темнеют.

        # 3. гауссов шум
        if self.noise > 0:
            noise = np.random.normal(0, self.noise, result_bgr.shape)
            # np.random.normal — генератор случайных чисел из нормального (гауссова) распределения.
            # Среднее = 0. Значит, случайные числа будут примерно поровну положительные и отрицательные — вокруг нуля
            # чем больше self.noise, тем сильнее разброс — тем заметнее зерно.
            # генерируется массив такой же формы, как картинка — для каждого пикселя каждого канала будет своё случайное число.
            # Итог: массив noise, заполненный случайными числами:

            result_bgr = result_bgr + noise # Прибавляем шум к каждому пикселю => зернистость
            # Каждый пиксель каждого канала получает своё случайное число. один и тот же пиксель изменяется по-разному в разных каналах.

        return np.clip(result_bgr, 0, 255).astype(np.uint8) # clip — обрезаем значения до диапазона [0, 255]. astype превращаем в целые числа