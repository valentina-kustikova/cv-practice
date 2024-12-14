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


def setSize(input_image):
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
        return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
    else:
        print("Область не выбрана. Пикселизация не выполнена.")
        return