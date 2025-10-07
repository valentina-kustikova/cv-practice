import cv2
import numpy as np


def change_resolution(image, new_width, new_height):

    if new_width <= 0 or new_height <= 0:
        raise ValueError("Ширина и высота должны быть >0")
    h, w = image.shape[:2]

    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    #Масштабируем исходное изображение
    scale_x = w / new_width
    scale_y = h / new_height
    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)
            #проверка границы
            src_x = min(src_x, w - 1)
            src_y = min(src_y, h - 1)
            resized_image[y, x] = image[src_y, src_x]
    return resized_image