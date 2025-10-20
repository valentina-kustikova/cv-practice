import cv2
import numpy as np


def sepia_filter(image, intensity=1.0):
    """
    Функция применения фотоэффекта сепии к изображению.
    """
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    sepia_matrix = sepia_matrix * intensity + np.eye(3) * (1 - intensity)
    sepia_image = cv2.transform(image, sepia_matrix)

    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return sepia_image


def apply_vignette_elliptical(image, scale=1.5, intensity=0.7):
    """
    Функция применения фотоэффекта виньетки к изображению.
    """
    height, width = image.shape[:2]

    kernel_x = cv2.getGaussianKernel(width, width / scale)
    kernel_y = cv2.getGaussianKernel(height, height / scale)
    kernel = kernel_y * kernel_x.T

    mask = kernel / kernel.max()
    mask = mask * intensity

    mask = mask[:, :, np.newaxis]

    result = image.astype(np.float32) * mask

    return np.clip(result, 0, 255).astype(np.uint8)


def add_rectangular_border(image, border_width=10, color=(0, 0, 0)):
    """
    Накладывает прямоугольную одноцветную рамку по краям изображения

    Args:
        image: исходное изображение
        border_width: ширина рамки в пикселях
        color: цвет рамки (B, G, R)

    Returns:
        изображение с рамкой
    """
    h, w = image.shape[:2]

    if border_width is None:
        border_width = int(min(h, w) / 10)

    image[0:border_width] = color
    image[-border_width:] = color
    image[border_width:-border_width, 0:border_width] = color
    image[border_width:-border_width, -border_width:] = color

    return image


def add_lens_flare(image, flare_position=(0.5, 0.5), intensity=1.0, flare_size=0.3):
    """
    Функция наложения эффекта одного большого блика объектива камеры.

    Args:
        image: исходное изображение
        flare_position: позиция блика в нормализованных координатах (0.0-1.0)
        intensity: интенсивность эффекта (0.0 - 1.0)
        flare_size: размер блика (0.1 - 1.0)
    """
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]

    # Конвертируем нормализованные координаты в пиксельные
    main_x = int(flare_position[0] * w)
    main_y = int(flare_position[1] * h)

    # Вычисляем радиус на основе размера
    base_radius = min(w, h) // 4
    radius = int(base_radius * flare_size)

    # Фиксированная яркость на основе интенсивности
    brightness = 0.4 * intensity

    # Создаем координатную сетку
    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - main_x) ** 2 + (y_coords - main_y) ** 2)

    # Создаем маску блика (гауссово распределение)
    flare_mask = np.exp(-(distance ** 2) / (2 * (radius ** 2)))
    flare_mask = flare_mask * brightness

    # Добавляем блик ко всем цветовым каналам
    for c in range(3):
        result[:, :, c] += flare_mask * 255

    # Обрезаем значения и возвращаем
    return np.clip(result, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, texture_intensity=0.3):
    """
    Накладывает текстуру акварельной бумаги

    Args:
        image: исходное изображение
        texture_intensity: интенсивность текстуры
    """
    h, w = image.shape[:2]

    # Создаем текстуру бумаги (зернистость + неравномерность)
    texture = np.random.normal(0, 0.1, (h, w)).astype(np.float32)

    # Добавляем низкочастотный шум для неравномерности
    low_freq_noise = cv2.resize(
        np.random.normal(0, 0.05, (h // 10, w // 10)),
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )
    texture += low_freq_noise

    # Применяем текстуру к изображению
    result = image.astype(np.float32)
    texture_3d = np.stack([texture] * 3, axis=2)

    # Смешиваем с оригиналом
    result = result * (1 - texture_intensity) + (result * (1 + texture_3d)) * texture_intensity

    # Добавляем легкое размытие для акварельного эффекта
    result = cv2.GaussianBlur(result.astype(np.float32), (3, 3), 0)

    return np.clip(result, 0, 255).astype(np.uint8)