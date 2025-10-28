import os

import cv2
import numpy as np


def sepia_filter(image, intensity=1.0):
    """
    Функция применения фотоэффекта сепии к изображению.

    Args:
        image: исходное изображение
        intensity: интенсивность сепии (0.0 - 1.0)
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

    Args:
        image: исходное изображение
        scale: масштаб виньетки
        intensity: интенсивность затемнения краев (0.0 - 1.0)
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
    Функция наложения эффекта блика.

    Args:
        image: исходное изображение
        flare_position: позиция блика в нормализованных координатах (0.0-1.0)
        intensity: интенсивность эффекта (0.0 - 1.0)
        flare_size: размер блика (0.1 - 1.0)
    """
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]

    main_x = int(flare_position[0] * w)
    main_y = int(flare_position[1] * h)

    base_radius = min(w, h) // 4
    radius = int(base_radius * flare_size)

    brightness = 0.4 * intensity

    y_coords, x_coords = np.ogrid[:h, :w]
    distance = np.sqrt((x_coords - main_x) ** 2 + (y_coords - main_y) ** 2)

    flare_mask = np.exp(-(distance ** 2) / (2 * (radius ** 2)))
    flare_mask = flare_mask * brightness

    for c in range(3):
        result[:, :, c] += flare_mask * 255

    return np.clip(result, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, texture_intensity=0.3):
    """
    Накладывает текстуру акварельной бумаги

    Args:
        image: исходное изображение
        texture_intensity: интенсивность текстуры (0.0 - 1.0)
    """
    h, w = image.shape[:2]

    texture = np.random.normal(0, 0.1, (h, w)).astype(np.float32)

    low_freq_noise = cv2.resize(
        np.random.normal(0, 0.05, (h // 10, w // 10)),
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )
    texture += low_freq_noise

    result = image.astype(np.float32)
    texture_3d = np.stack([texture] * 3, axis=2)

    result = result * (1 - texture_intensity) + (result * (1 + texture_3d)) * texture_intensity

    result = cv2.GaussianBlur(result.astype(np.float32), (3, 3), 0)

    return np.clip(result, 0, 255).astype(np.uint8)


def draw_green_frame(image, x1, y1, x2, y2):
    """
    Рисует зеленую рамку толщиной 1 пиксель по заданным координатам

    Args:
        image: исходное изображение
        x1, y1: координаты левого верхнего угла
        x2, y2: координаты правого нижнего угла
    """
    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0 and x2 > x1 and y2 > y1:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image


def pixelate_region(image, x1, y1, x2, y2, pixel_size=10):
    """
    Пикселизация заданной прямоугольной области изображения

    Args:
        image: исходное изображение
        x1, y1: координаты левого верхнего угла области
        x2, y2: координаты правого нижнего угла области
        pixel_size: размер пикселя для пикселизации
    """

    if (x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or
            x2 > image.shape[1] or y2 > image.shape[0]):
        return image

    region = image[y1:y2, x1:x2]

    h, w = region.shape[:2]

    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    image[y1:y2, x1:x2] = pixelated

    return image


def apply_figure_frame(image, frame_path):
    """
    Наложение фигурной рамки по краям изображения

    Args:
        image: исходное изображение
        frame_path: путь до рамки
    """
    result = image.copy()
    h, w = result.shape[:2]

    frame_path = os.path.join('frames', frame_path)

    if os.path.exists(frame_path):
        frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if frame_img is not None:
            frame_resized = cv2.resize(frame_img, (w, h))

            frame_bgr = frame_resized[:, :, :3]
            alpha = frame_resized[:, :, 3] / 255.0

            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - alpha) + frame_bgr[:, :, c] * alpha

    return result