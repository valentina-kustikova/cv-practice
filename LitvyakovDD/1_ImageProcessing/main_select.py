from pathlib import Path
import cv2
import numpy as np

from inner.src import ImageFilter, check_image

IMAGE_DIR = Path(__file__).parent / "img"
RESULT_DIR = Path(__file__).parent / "result"

def read_image(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось открыть изображение.")
    return image

def write_image(filename, image):
    extension = filename.suffix
    status, data = cv2.imencode(extension, image)
    if not status:
        raise ValueError("Не удалось сохранить изображение.")
    filename.parent.mkdir(exist_ok=True)
    data.tofile(filename)

def main():
    image_name = input("Введите имя изображения: ").strip()
    image_path = IMAGE_DIR / image_name
    try:
        image = read_image(image_path)
        check_image(image)
    except (OSError, ValueError, cv2.error) as error:
        print("Не удалось открыть изображение:", error)
        return 1

    filter_names = [
        "resize",
        "grayscale",
        "antique",
        "fade",
        "film",
        "matte",
        "scratches",
        "neon",
    ]
    while True:
        print("\nВыберите фильтр:")
        for number, name in enumerate(filter_names, 1):
            print(f"{number}. {name}")
        print("0. Выход")
        choice = input("Ваш выбор: ").strip()
        try:
            number = int(choice)
        except ValueError:
            print("Введите номер фильтра или 0 для выхода.")
            continue
        if number == 0:
            return 0
        if number < 1 or number > len(filter_names):
            print("Неверный номер фильтра.")
            continue

        filter_name = filter_names[number - 1]
        try:
            parameters = {}
            if filter_name == "resize":
                parameters["width"] = int(input("Новая ширина: "))
                parameters["height"] = int(input("Новая высота: "))
            elif filter_name == "scratches":
                parameters["count"] = int(input("Количество царапин: "))
            elif filter_name == "antique":
                parameters["strength"] = float(input("Сила эффекта (0..1): "))
            elif filter_name == "fade":
                parameters["strength"] = float(input("Сила эффекта (0..1): "))
            elif filter_name == "film":
                parameters["strength"] = float(input("Сила эффекта (0..1): "))
            elif filter_name == "matte":
                parameters["strength"] = float(input("Сила эффекта (0..1): "))
            elif filter_name == "neon":
                parameters["threshold"] = float(input("Порог контуров: "))
                parameters["strength"] = float(input("Сила эффекта (0..1): "))
                parameters["brightness"] = float(input("Яркость фона (0..1): "))

            filter_object = ImageFilter.get_filter(filter_name, **parameters)
            result = filter_object.apply_filter(image)
            output_path = RESULT_DIR / f"{image_path.stem}_{number}{image_path.suffix}"
            write_image(output_path, result)
        except (OSError, ValueError, TypeError, cv2.error) as error:
            print("Не удалось применить фильтр:", error)
            continue
        print("Результат сохранён:", output_path)

if __name__ == "__main__":
    raise SystemExit(main())
