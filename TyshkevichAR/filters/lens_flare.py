import numpy as np
import math

def apply_lens_flare(image, intensity=0.7):

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    pos_x = 3 * w // 4 #позиция и радиус
    pos_y = h // 3
    flare_radius = min(h, w) // 2

    for y in range(max(0, pos_y - flare_radius), min(h, pos_y + flare_radius)):
        for x in range(max(0, pos_x - flare_radius), min(w, pos_x + flare_radius)):
            # Расстояние от центра блика
            dist = math.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)

            if dist <= flare_radius:
                flare_intensity = intensity * (1 - dist / flare_radius)
                for channel in range(3):
                    result[y, x, channel] = min(
                        255,
                        result[y, x, channel] + 255 * flare_intensity #засвечиваем
                    )

    return np.clip(result, 0, 255).astype(np.uint8)