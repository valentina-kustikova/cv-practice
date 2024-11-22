import argparse
import image_operations
import image_filters
import cv2

# Глобальные переменные для выделения области
ref_point = []
cropping = False


def select_roi(event, x, y, flags, param):
    """
    Обработчик событий мыши для выделения области.
    """
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Нарисовать выделенную область
        cv2.rectangle(param, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Image", param)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Программа для обработки изображений с фильтрами.")
    parser.add_argument("-i", "--input", required=True, help="Путь к входному изображению.")
    parser.add_argument("-o", "--output", required=True, help="Путь для сохранения обработанного изображения.")
    parser.add_argument("-f", "--filter", required=True,
                        choices=["grayscale", "vignette", "pixelate", "resize", "sepia"],
                        help="Выберите фильтр: grayscale, vignette, pixelate, resize или sepia.")

    parser.add_argument("--radius", type=float, default=500,
                        help="Радиус эффекта виньетки (по умолчанию: 500)")
    parser.add_argument("--intensity", type=float, default=1.0,
                        help="Интенсивность эффекта виньетки (по умолчанию: 1.0)")

    parser.add_argument("--pixel_size", type=int, default=10,
                        help="Размер пикселей для эффекта пикселизации (по умолчанию: 10)")

    parser.add_argument("--resize_width", type=int,
                        help="Ширина для изменения размера изображения")

    parser.add_argument("--resize_height", type=int,
                        help="Высота для изменения размера изображения")

    return parser.parse_args()


def main():
    global processed_image
    args = parse_arguments()

    # Загрузка изображения
    input_image = image_operations.load_image(args.input)
    if input_image is None:
        print("Ошибка: не удалось загрузить изображение. Проверьте путь.")
        return

    # Применение фильтра
    if args.filter == "grayscale":
        processed_image = image_filters.apply_grayscale(input_image)
    elif args.filter == "vignette":
        processed_image = image_filters.apply_vignette(input_image, args.radius, args.intensity)
    elif args.filter == "pixelate":
        print("Выберите область для пикселизации мышкой и нажмите 'c' для подтверждения или 'r' для сброса.")
        clone = input_image.copy()
        cv2.imshow("Image", input_image)
        cv2.setMouseCallback("Image", select_roi, clone)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                clone = input_image.copy()
                cv2.imshow("Image", clone)

            elif key == ord("c"):
                break

        cv2.destroyAllWindows()

        # Если область выбрана
        if len(ref_point) == 2:
            # Получаем координаты выбранной области
            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]
            rect = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            # Применяем пикселизацию только к этой области
            processed_image = image_filters.apply_pixelation(input_image, rect, args.pixel_size)
        else:
            print("Область не выбрана. Пикселизация не выполнена.")
            return

    elif args.filter == "resize":
        if args.resize_width is None or args.resize_height is None:
            print("Ошибка: для изменения размера необходимо указать и ширину, и высоту.")
            return
        processed_image = image_filters.resize_image(input_image, args.resize_width, args.resize_height)
    elif args.filter == "sepia":
        processed_image = image_filters.apply_sepia(input_image)

    # Сохранение результата
    success = image_operations.save_image(args.output, processed_image)
    if success:
        print(f"Изображение успешно сохранено: {args.output}")
    else:
        print("Ошибка: не удалось сохранить изображение.")


if __name__ == "__main__":
    main()
