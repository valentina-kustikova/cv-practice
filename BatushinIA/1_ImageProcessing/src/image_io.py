import os
import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """
    Читает изображение и возвращает матрицу интенсивностей.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    return image


def save_image(path: str, image: np.ndarray) -> None:
    """Сохраняет изображение"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not cv2.imwrite(path, image):
        raise IOError(f"Не удалось сохранить изображение: {path}")