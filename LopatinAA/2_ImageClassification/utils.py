
import os

import cv2
import numpy as np

# Допустимые расширения файлов
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def to_system_path(path):
    """Для обхода ограничения на максимальную длину пути Windows."""
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    return "\\\\?\\" + os.path.abspath(path)


def read_image(path):
    system_path = to_system_path(path)
    if not os.path.isfile(system_path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    data = np.fromfile(system_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return image


def is_image_file(file_name):
    """Проверяет по расширению, является ли файл изображением."""
    extension = os.path.splitext(file_name)[1].lower()
    return extension in IMAGE_EXTENSIONS


def list_image_files(directory):
    """Возвращает отсортированный список путей к изображениям в директории."""
    paths = []
    for file_name in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, file_name)
        if os.path.isfile(to_system_path(full_path)) and is_image_file(file_name):
            paths.append(full_path)
    return paths
