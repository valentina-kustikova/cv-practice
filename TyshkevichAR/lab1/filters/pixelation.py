import numpy as np


def apply_pixelation(image, region):

    x1, y1, x2, y2 = region
    block_size = 15  # размер блока пикселей
    result = image.copy()

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    for y in range(y1, y2, block_size):
        for x in range(x1, x2, block_size):
            # Границы блока
            block_y_end = min(y + block_size, y2)
            block_x_end = min(x + block_size, x2)
            block = image[y:block_y_end, x:block_x_end]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8) # Средний цвет
                result[y:block_y_end, x:block_x_end] = avg_color

    return result