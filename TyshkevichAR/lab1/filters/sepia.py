#sepia
import numpy as np

def apply_sepia(image):
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    # матричное преобразование
    sepia_image = np.dot(image.astype(np.float32), sepia_matrix.T)

    return np.clip(sepia_image, 0, 255).astype(np.uint8)