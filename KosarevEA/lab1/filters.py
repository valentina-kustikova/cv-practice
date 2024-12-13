import numpy as np
import cv2

def convert_to_grayscale(image):
    if image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b

    return grayscale.astype(np.uint8)


# def resize_image(image, new_width, new_height):
#     # Получаем исходные размеры изображения
#     original_height, original_width = image.shape[:2]
#     # Создаем новое изображение с заданными размерами
#     resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
#     # Рассчитываем коэффициенты масштабирования
#     x_ratio = original_width / new_width
#     y_ratio = original_height / new_height
    
#     for i in range(new_height):
#         for j in range(new_width):
#             # Находим соответствующие координаты в исходном изображении
#             src_x = int(j * x_ratio)
#             src_y = int(i * y_ratio)
#             resized_image[i, j] = image[src_y, src_x]
    
#     return resized_image

def resize_image(image, new_width, new_height):
    if new_width <= 0 or new_height <= 0:
        raise ValueError('Width and height must be greater than zero')  
    
    height, width, nchannels = image.shape
    
   
    x_scale = width / new_width
    y_scale = height / new_height

    x_ind = np.floor(np.arange(new_width) * x_scale).astype(int)
    y_ind = np.floor(np.arange(new_height) * y_scale).astype(int)

    x_ind = np.clip(x_ind, 0, width - 1)
    y_ind = np.clip(y_ind, 0, height - 1)

    result_image = image[y_ind[:, None], x_ind]

    return result_image


def apply_sepia(image):
    # Проверяем, что изображение имеет 3 канала (RGB)
    if image.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    
    # Извлекаем отдельные каналы изображения
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    # Применяем сепию к каждому каналу
    sepia_r = (0.272 * r) + (0.534 * g) + (0.131 * b)
    sepia_g = (0.349 * r) + (0.686 * g) + (0.168 * b)
    sepia_b = (0.393 * r) + (0.769 * g) + (0.189 * b)

    sepia_image = np.stack((sepia_r, sepia_g, sepia_b), axis=-1)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return sepia_image


def apply_vignette(image, radius=1.5, intensity=1.0):
    rows, cols = image.shape[:2]
    
    mask = np.zeros((rows, cols), np.float32)

    # Определяем центр изображения
    center_x = cols // 2
    center_y = rows // 2

    # Генерация градиента виньетки
    for y in range(rows):
        for x in range(cols):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask[y, x] = 1 - min(distance / radius, 1)

    # Normalize the mask and apply intensity
    mask = mask / mask.max()
    mask = mask ** intensity

    vignette_image = np.copy(image)

   # Применяем эффект виньетки к каждому цветному каналу
    for i in range(3):  
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask
    
    return vignette_image.astype(np.uint8)


def pixelate_region(image, rect, pixel_size):
    x, y, w, h = rect
    pixelated_image = np.copy(image)
    # Перебор блоков по высоте и ширине
    for i in range(y, y + h, pixel_size):
        for j in range(x, x + w, pixel_size):
            # Извлечение блока
            block = pixelated_image[i:i+pixel_size, j:j+pixel_size]
            if block.size == 0:
                continue
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            pixelated_image[i:i+pixel_size, j:j+pixel_size] = avg_color
    return pixelated_image