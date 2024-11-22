import numpy as np


def apply_grayscale(image):
    """Конвертирует изображение в оттенки серого."""
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def apply_vignette(image, radius=500, intensity=1.0):
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

    mask = mask / mask.max()
    mask = mask ** intensity

    vignette_image = np.copy(image)

    # Применяем эффект виньетки к каждому цветному каналу
    for i in range(3):
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask

    return vignette_image.astype(np.uint8)


def apply_pixelation(image, rect, pixel_size):
    """Применяет эффект пикселизации к указанной области изображения."""
    x, y, w, h = rect
    pixelated_image = image.copy()
    for i in range(y, y + h, pixel_size):
        for j in range(x, x + w, pixel_size):
            pixel_block = pixelated_image[i:i + pixel_size, j:j + pixel_size]
            if pixel_block.size == 0:
                continue
            average_color = pixel_block.mean(axis=(0, 1)).astype(np.uint8)
            pixelated_image[i:i + pixel_size, j:j + pixel_size] = average_color
    return pixelated_image


def resize_image(image, new_width, new_height):
    """Изменяет размер изображения."""
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
    """Применяет сепию к изображению."""
    # Коэффициенты для преобразования в сепию
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    # Преобразование изображения в массив с плавающей точкой
    image_float = image.astype(np.float32)

    # Применение матрицы сепии к изображению
    sepia_image = np.dot(image_float, sepia_matrix.T)

    # Ограничение значений в диапазоне [0, 255]
    sepia_image = np.clip(sepia_image, 0, 255)

    # Преобразование обратно в целочисленный тип
    return sepia_image.astype(np.uint8)
