import numpy as np
import random


def apply_watercolor_paper(image):
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]

    for y in range(h):
        for x in range(w):
            for c in range(3):
                noise = random.uniform(-30, 30) #накладываем шум в этом диапазоне
                result[y, x, c] = np.clip(result[y, x, c] + noise, 0, 255)
    return result.astype(np.uint8)