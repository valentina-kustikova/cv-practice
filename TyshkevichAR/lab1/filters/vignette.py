#vignette
import numpy as np

def apply_vignette(image, intensity=0.8):
    h, w = image.shape[:2]

    y_coords, x_coords = np.indices((h, w))
    center_x, center_y = w // 2, h // 2
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

    # расстояния от центра до угла
    dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    norm_dist = dist / max_dist

    # Создаем виньетку
    darken = 1.0 - intensity * norm_dist
    vignette = np.maximum(darken, 0.2)

    # Применяем виньетку ко всем каналам
    vignette_image = image.astype(np.float32) * vignette[:, :, np.newaxis]

    return np.clip(vignette_image, 0, 255).astype(np.uint8)