import numpy as np
from .base import ImageFilter


class Neon(ImageFilter):
    # Неон поверх оригинала по контурам и чуть вокруг них

    def __init__(self, strength: float = 9.0,
                 threshold: float = 20.0,
                 glow_size: float = None,
                 glow_strength: float = None,
                 edge_softness: float = None):
        # strength — яркость 
        # threshold — порог: контуры слабее порога не рисуются
        if strength <= 0:
            raise ValueError('strength должен быть > 0')
        if threshold < 0:
            raise ValueError('threshold должен быть >= 0')
        self.strength = strength
        self.threshold = threshold

    def _sobel(self, gray):
        # Оператор Собеля - распознает границы на серых картинках
        # дополняем картинку по 1 пикселю с каждой стороны, копируя края тк Sobel использует окно 3×3. 
        # Без padding крайние пиксели не имели бы соседей — формула не работает.
        g = np.pad(gray, 1, mode='edge')

        # свёртка с ядром Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] — производная по X
        # изменение яркости по X — вертикальные границы дают большое gx
        gx = (-1 * g[:-2, :-2] + 1 * g[:-2, 2:]
              - 2 * g[1:-1, :-2] + 2 * g[1:-1, 2:]
              - 1 * g[2:, :-2] + 1 * g[2:, 2:])

        # Ядро Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] — производная по Y
        gy = (-1 * g[:-2, :-2] - 2 * g[:-2, 1:-1] - 1 * g[:-2, 2:]
              + 1 * g[2:, :-2] + 2 * g[2:, 1:-1] + 1 * g[2:, 2:])

        # объединяет два направления в одно число — общую силу границы.
        # итог: magnitude — матрица силы границ
        return np.sqrt(gx ** 2 + gy ** 2)

    def apply_filter(self, image):
        # 1. Серое 
        if len(image.shape) == 3: # если картинка цветная переводим в серую
            gray = (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]).astype(np.float32)
            # Формула яркости в цветовом пространстве YUV: Y — яркость, U и V — цветовая информация.
            # берём только яркость и используем её как серое изображение, а U и V отбрасываем.
                    
        else:
            gray = image.astype(np.float32)

        # 2.  применяем оператор Собеля
        magnitude = self._sobel(gray)
        #  Магнитуда — матрица, где для каждого пикселя записано: «В этой точке яркость не меняется» (0)
        # «В этой точке яркость резко меняется»

        # 3. Усиление через strength. что это меняет: яркость контуров
        magnitude = magnitude * self.strength

        # 4. Порог: слабые контуры обнуляем. np.where(условие, значение_если_True, значение_если_False)
        magnitude = np.where(magnitude > self.threshold, magnitude, 0.0)

        # 5. Обрезаем до 255 и превращаем в целые числа
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        # 6. Наложение на оригинал. cерое фото + бирюзовые полосы в местах магнитуды
        # Прибавление контура к B и G каналам
        res = image.astype(np.float32) # берём оригинал — копируем в res
        res[:, :, 0] += magnitude   #  к каналу B (синий) прибавляем magnitude поэлементно.
        res[:, :, 1] += magnitude   
        # Почему именно B и G: B + G = бирюзовый 
       
        return np.clip(res, 0, 255).astype(np.uint8) # обрезает значения до допустимых и превращает в целые числа