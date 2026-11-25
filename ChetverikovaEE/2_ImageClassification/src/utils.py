from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Настроить логирование проекта (формат: время, уровень, имя модуля, сообщение).

    :param level: уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    :return: логгер проекта с именем 'src'
    """
    level_value = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=level_value,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return logging.getLogger("src")


def set_seed(seed: int) -> None:
    """Зафиксировать зерно генераторов случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def read_image(path: PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Прочитать изображение с диска и вернуть массив BGR (np.ndarray).

    :param path: путь к файлу изображения
    :param flags: флаги cv2 (по умолчанию — цветное изображение)
    :raises FileNotFoundError: если файла нет
    :raises IOError: если файл не удалось декодировать
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Файл изображения не найден: {image_path.resolve()}")
    data = image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None:
        raise IOError(
            f"Не удалось декодировать изображение (файл повреждён или формат не поддерживается): {image_path}"
        )
    return image


def save_image(path: PathLike, image: np.ndarray) -> None:
    """Сохранить изображение на диск (аналогично read_image, поддерживает кириллические пути)."""
    image_path = Path(path)
    extension = image_path.suffix or ".png"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise IOError(f"Не удалось закодировать изображение для сохранения: {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(buffer.tobytes())


def load_sample(source: Union[PathLike, np.ndarray]) -> np.ndarray:
    if isinstance(source, (str, Path)):
        return read_image(source)
    return source


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Перевести изображение в градации серого (поддерживает BGR, BGRA и уже серое)."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ensure_dir(path: PathLike) -> Path:
    """Создать каталог (включая родительские), если он не существует; вернуть Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
