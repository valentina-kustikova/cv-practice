import cv2
from pathlib import Path


def read_image(path):
    """Загружает изображение и сообщает об ошибке, если открыть его не удалось."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    return image


def save_image(image, path):
    """Сохраняет обработанное изображение."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise ValueError(f"Не удалось сохранить изображение: {output_path}")
