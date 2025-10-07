import numpy as np
import math

def apply_vignette(image, intensity=0.8):
    h, w = image.shape[:2]

    vignette = np.zeros((h, w), dtype=np.float32)
    center_x, center_y = w // 2, h // 2
    #расстояние от центра до угла
    max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

    for y in range(h):
        for x in range(w):
            dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)  # центр
            # нормализуем
            norm_dist = dist / max_dist
            darken = 1.0 - intensity * norm_dist
            vignette[y, x] = max(darken, 0.2) #min яркость
    # Применяем к каждому каналу
    vignette_image = np.zeros_like(image, dtype=np.float32)

    for channel in range(3):
        vignette_image[:, :, channel] = image[:, :, channel].astype(np.float32) * vignette
    return np.clip(vignette_image, 0, 255).astype(np.uint8)