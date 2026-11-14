"""Вспомогательные функции для работы с изображениями."""

from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Читает изображение с диска средствами OpenCV.

    cv2.imread на Windows не открывает пути с кириллицей, поэтому файл
    читается как массив байт и декодируется через cv2.imdecode.

    :param path: путь к файлу изображения.
    :param flags: флаг декодирования OpenCV (по умолчанию цветное BGR-изображение).
    :return: изображение в формате np.ndarray (H x W x 3, BGR, uint8).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return image


def resize_keep_aspect(image, target_size, interpolation=cv2.INTER_LINEAR):
    """Масштабирует изображение так, чтобы большая сторона стала равна target_size."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    if scale == 1.0:
        return image
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, new_size, interpolation=interpolation)
