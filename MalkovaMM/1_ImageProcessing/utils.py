import cv2
from pathlib import Path


def read_image(image_path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {image_path} не найден")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение {image_path}")
    return image