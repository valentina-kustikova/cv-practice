# Состаренная фотография: текстура + шум

import os
import numpy as np
import cv2
from .base import ImageFilter


class Aged(ImageFilter):

    def __init__(self, texture_path=None,
                 noise_level=10.0,
                 scratch_intensity=0.6,
                 paper_color=(20, 25, 35)):
        # texture_path — путь к PNG-текстуре 
        # noise_level — уровень шума 
        # scratch_intensity — заметность текстуры (0..1)
        # paper_color — цвет бумаги в BGR (куда тянется тёмная текстура). в формулах смешивания тёмные места текстуры подменяются на этот цвет

        # Проверка: путь к текстуре обязателен
        if texture_path is None:
            raise ValueError('texture_path обязателен для aged')
        # Проверка: файл текстуры существует
        if not os.path.exists(texture_path):
            raise FileNotFoundError(f'Текстура не найдена: {texture_path}')

        self.texture_path = texture_path
        self.noise_level = noise_level
        self.scratch_intensity = scratch_intensity
        # Преобразуем цвет бумаги в numpy-массив float32
        self.paper_color = np.array(paper_color, dtype=np.float32)

    def _apply_texture(self, img):
        # Наложение текстуры через np.tile. берёт массив и повторяет его N раз по указанным осям.

        # Читаем текстуру
        tex = cv2.imread(self.texture_path, cv2.IMREAD_UNCHANGED) # читает как есть — сохраняет все 4 канала (BGR + альфа(прозрачность))
        if tex is None:
            raise ValueError(f'Не удалось прочитать текстуру: {self.texture_path}')

        # Размеры картинки и текстуры
        h, w = img.shape[:2]
        tex_h, tex_w = tex.shape[:2]

        # Сколько раз нужно повторить текстуру по вертикали и горизонтали
        reps_y = h // tex_h + 1
        reps_x = w // tex_w + 1

        # Размножение текстуры по сетке (np.tile)
        if tex.ndim == 2:
            # берёт массив и повторяет его N раз по указанным осям. тут без оси каналов
            tex = np.tile(tex, (reps_y, reps_x))
        else:
            # RGB или RGBA — с осью каналов
            tex = np.tile(tex, (reps_y, reps_x, 1))
        # Обрезаем текстуру до размера картинки
        tex = tex[:h, :w]

        # Приводим картинку к float32
        img_f = img.astype(np.float32)
        k = self.scratch_intensity

       
        # превращает текстуру из целых чисел в дробные. / 255.0 нормализует текстуру к [0, 1]. добавляем новую ось в конец.
        tex_norm = (tex.astype(np.float32) / 255.0)[..., None] # tex_norm — матрица формы (h, w, 1), где: 0 — тёмная часть текстуры, 1 — светлая 
        
        # Смешиваем: где текстура тёмная — тянем к paper_color
        # mix — это степень замены пикселя на заданный цвет бумаги: где текстура тёмная — mix близко к k (сильная замена), 
        # где светлая — к нулю, а формула img * (1 - mix) + paper_color * mix взвешивает оригинал и paper_color по этой степени.
        mix = (1.0 - tex_norm) * k # тк мы хотим, чтобы тёмное в текстуре давало сильную замену
        return img_f * (1.0 - mix) + self.paper_color * mix # Формула взвешенно смешивает оригинал и цвет бумаги: 
            # где mix = 0 — остаётся оригинал, где mix = 1 — полностью заданный цвет бумаги.


    def apply_filter(self, image):
        # Приводим картинку к float32
        img = image.astype(np.float32)

        # 1. Наложение текстуры (сетка + размножение). вызов приватного метода _apply_texture, который накладывает текстуру на фото.
        img = self._apply_texture(img)

        # 2. гауссов шум
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, img.shape)
            img = img + noise
         # np.random.normal — генератор случайных чисел из нормального (гауссова) распределения.
         # Среднее = 0. Значит, случайные числа будут примерно поровну положительные и отрицательные — вокруг нуля
         # чем больше self.noise, тем сильнее разброс — тем заметнее зерно.
         # генерируется массив такой же формы, как картинка — для каждого пикселя каждого канала будет своё случайное число.
         # Итог: массив noise, заполненный случайными числами
         # Прибавляем шум к каждому пикселю => зернистость
         # Каждый пиксель каждого канала получает своё случайное число. один и тот же пиксель изменяется по-разному в разных каналах.


        # Обрезаем значения в [0, 255] и возвращаем uint8
        return np.clip(img, 0, 255).astype(np.uint8)