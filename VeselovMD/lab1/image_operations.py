import cv2


def load_image(file_path):
    """Загружает изображение с указанного пути."""
    return cv2.imread(file_path)


def save_image(file_path, image):
    """Сохраняет изображение на указанный путь."""
    try:
        cv2.imwrite(file_path, image)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
        return False
