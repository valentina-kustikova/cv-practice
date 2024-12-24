from typing import BinaryIO

import cv2
import numpy as np

def grayscale(img):
    if img.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    result = 0.299 * r + 0.587 * g + 0.114 * b
    return result.astype(np.uint8)



def resize(img, value):
    
    hght, wdth, nchannels = img.shape
    new_hght = int(hght * value)
    new_wdth = int(wdth * value)
    result = np.zeros((new_hght, new_wdth, nchannels),dtype=np.uint8)

    for i in range (new_hght):
        for j in range(new_wdth):
            result[i, j] = img[int(i/value), int(j/value)]
    
    return result

        



def sepia(img):
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    result[:, :, 0] = np.clip(0.272 * R + 0.534 * G + 0.131 * B, 0, 255)
    result[:, :, 1] = np.clip(0.349 * R + 0.686 * G + 0.168 * B, 0, 255)
    result[:, :, 2] = np.clip(0.393 * R + 0.769 * G + 0.189 * B, 0, 255)

    return result

def vignette(img, radius):
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    height, width, nchannels = img.shape
    x = img.shape[1] // 2
    y = img.shape[0] // 2

    y_val, x_val = np.indices((height, width)) #две матрицы

    distance = np.sqrt((x_val - x) ** 2 + (y_val - y) ** 2)
    coef = np.clip(1 - distance / radius, 0, 1)
    result = (img * coef[:, :, np.newaxis]).astype(np.uint8)

    return result


def pixelate(img, pixel_size, region):
    if region is None:
        print('no region selected')
        return None

    x, y , w, h = region
    result = np.copy(img)

    for i in range(y, y + h, pixel_size):
        for j in range(x, x + w, pixel_size):
            # Извлечение блока
            block = result[i:i + pixel_size, j:j + pixel_size]
            if block.size == 0:
                continue
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            result[i:i + pixel_size, j:j + pixel_size] = avg_color
    return result