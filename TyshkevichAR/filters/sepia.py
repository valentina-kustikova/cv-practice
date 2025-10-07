import numpy as np


def apply_sepia(image):
    sepia_matrix = np.array([  # Матрица преобразования для сепии
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_image = np.zeros_like(image, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x].astype(np.float32)
            sepia_pixel = np.dot(sepia_matrix, pixel)
            sepia_image[y, x] = np.clip(sepia_pixel, 0, 255) # Ограничиваем до 255
    return sepia_image.astype(np.uint8)