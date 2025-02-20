import numpy as np
from typing import Tuple


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts RGB image to grayscale"""

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b

    return grayscale.astype(np.uint8)


def resize_image(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Resizes image to specified width and height"""
    
    height, width, nchannels = image.shape
    
    x_scale = width / new_width
    y_scale = height / new_height

    x_ind = np.floor(np.arange(new_width) * x_scale).astype(int)
    y_ind = np.floor(np.arange(new_height) * y_scale).astype(int)

    x_ind = np.clip(x_ind, 0, width - 1)
    y_ind = np.clip(y_ind, 0, height - 1)

    result_image = image[y_ind[:, None], x_ind]

    return result_image


def apply_sepia(image: np.ndarray) -> np.ndarray:
    """Applies a sepia filter to RGB image"""
    
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    sepia_r = (0.272 * r) + (0.534 * g) + (0.131 * b)
    sepia_g = (0.349 * r) + (0.686 * g) + (0.168 * b)
    sepia_b = (0.393 * r) + (0.769 * g) + (0.189 * b)

    sepia_image = np.stack((sepia_r, sepia_g, sepia_b), axis=-1)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return sepia_image


def apply_vignette(image: np.ndarray, radius: float = 1.5, intensity: float = 1.0) -> np.ndarray:
    """Applies a vignette effect to image with given radius and intensity"""
    rows, cols = image.shape[:2]
    
    mask = np.zeros((rows, cols), np.float32)

    center_x = cols // 2
    center_y = rows // 2

    for y in range(rows):
        for x in range(cols):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask[y, x] = 1 - min(distance / radius, 1)

    mask = mask / mask.max()
    mask = mask ** intensity

    vignette_image = np.copy(image)

    for i in range(3):  
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask
    
    return vignette_image.astype(np.uint8)


def pixelate_region(image: np.ndarray, rect: Tuple[int, int, int, int], pixel_size: int) -> np.ndarray:
    """Applies a pixelation effect to specified rectangular region of image"""

    x, y, w, h = rect
    pixelated_image = np.copy(image)

    for i in range(y, y + h, pixel_size):
        for j in range(x, x + w, pixel_size):
            block = pixelated_image[i:i+pixel_size, j:j+pixel_size]
            if block.size == 0:
                continue
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            pixelated_image[i:i+pixel_size, j:j+pixel_size] = avg_color

    return pixelated_image