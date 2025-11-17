#simple_border
import numpy as np

def apply_simple_border(image, border_width=20):
    h, w = image.shape[:2]

    # Создаем новое изображение с рамкой
    bordered_image = np.zeros((h + 2 * border_width, w + 2 * border_width, 3), dtype=np.uint8)
    bordered_image[:, :] = [0, 0, 0]
    #изображение в центр
    bordered_image[border_width:border_width + h, border_width:border_width + w] = image

    return bordered_image