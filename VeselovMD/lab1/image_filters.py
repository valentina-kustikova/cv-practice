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

    # Normalize the mask and apply intensity
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


def resize_image(image, width, height):
    """Изменяет размер изображения."""
    resized_image = np.zeros((height, width, 3), dtype=image.dtype)
    scale_x, scale_y = image.shape[1] / width, image.shape[0] / height
    for i in range(height):
        for j in range(width):
            src_x, src_y = int(j * scale_x), int(i * scale_y)
            resized_image[i, j] = image[src_y, src_x]
    return resized_image


def apply_sepia(image):
    """Применяет сепию к изображению."""
    sepia_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            tr = 0.272 * r + 0.534 * g + 0.131 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.393 * r + 0.769 * g + 0.189 * b
            sepia_image[i, j] = [min(tr, 255), min(tg, 255), min(tb, 255)]
    return sepia_image.astype(np.uint8)
