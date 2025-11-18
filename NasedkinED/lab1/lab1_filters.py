import argparse

import cv2
import numpy as np

# Глобальные переменные для выделения области мышью
ref_point = []
cropping = False
image_for_selection = None


def mouse_callback(event, x, y, flags, param):
    global ref_point, cropping, image_for_selection

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_img = image_for_selection.copy()
            cv2.rectangle(temp_img, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Region", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        temp_img = image_for_selection.copy()
        cv2.rectangle(temp_img, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Select Region", temp_img)



def resize_image(image, new_width, new_height):
    """
    Изменение размера методом ближайшего соседа.
    """
    height, width, channels = image.shape

    # Генерируем сетку координат
    # linspace создает равномерно распределенные значения индексов
    x_indices = np.linspace(0, width - 1, new_width).astype(int)
    y_indices = np.linspace(0, height - 1, new_height).astype(int)

    # Используем meshgrid или broadcasting для создания 2D массива индексов
    # image[y, x] где y - вектор строк, x - вектор столбцов
    resized_image = image[y_indices[:, None], x_indices]

    return resized_image


def apply_sepia(image):
    """
    Применение сепии с помощью матричного умножения.
    """
    # Коэффициенты для BGR (OpenCV использует BGR, формула обычно для RGB)
    # Преобразование: B_new = 0.272R + 0.534G + 0.131B и т.д.
    # Матрица для умножения (транспонированная для image @ matrix.T или через dot)
    # Порядок каналов в image: B, G, R.

    # Матрица коэффициентов для умножения справа: [B, G, R]
    # Строки матрицы соответствуют входным каналам (B, G, R), столбцы - выходным
    kernel = np.array([
        [0.131, 0.534, 0.272],  # Blue coeff
        [0.168, 0.686, 0.349],  # Green coeff
        [0.189, 0.769, 0.393]  # Red coeff
    ]).T  # Транспонируем для корректного умножения

    # Векторное умножение
    sepia_image = image.dot(kernel)

    # Клиппинг значений и приведение к uint8
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image


def apply_vignette(image, strength=0.5):
    """
    Виньетка с использованием meshgrid для генерации маски расстояний.
    """
    height, width, _ = image.shape
    center_y, center_x = height / 2, width / 2

    # Создаем сетку координат
    Y, X = np.ogrid[:height, :width]

    # Считаем расстояние от центра для всех точек сразу
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    max_dist_sq = center_x ** 2 + center_y ** 2

    # Нормализованное расстояние
    norm_dist = np.sqrt(dist_sq) / np.sqrt(max_dist_sq)

    # Фактор затемнения
    factor = 1 - strength * (norm_dist ** 2)
    factor = np.clip(factor, 0, 1)

    # Применяем фактор ко всем каналам (добавляем ось для broadcasting)
    vignette_image = (image * factor[..., np.newaxis]).astype(np.uint8)

    return vignette_image


def pixelate_region(image, x1, y1, x2, y2, pixel_size=10):
    """
    Пикселизация области с использованием numpy slicing.
    """
    # Нормализация координат (чтобы x1 < x2)
    x_start, x_end = min(x1, x2), max(x1, x2)
    y_start, y_end = min(y1, y2), max(y1, y2)

    # Защита от выхода за границы
    h_img, w_img = image.shape[:2]
    x_start, y_start = max(0, x_start), max(0, y_start)
    x_end, y_end = min(w_img, x_end), min(h_img, y_end)

    roi_h = y_end - y_start
    roi_w = x_end - x_start

    if roi_h <= 0 or roi_w <= 0:
        return image

    roi = image[y_start:y_end, x_start:x_end]

    # "Сжатие": берем каждый n-й пиксель
    small = roi[::pixel_size, ::pixel_size]

    # "Растягивание" обратно: повторяем пиксели
    # np.repeat повторяет элементы массива
    expanded_h = np.repeat(small, pixel_size, axis=0)
    expanded = np.repeat(expanded_h, pixel_size, axis=1)

    # Обрезаем до оригинального размера ROI (на случай некратности размеров)
    expanded = expanded[:roi_h, :roi_w]

    result = image.copy()
    result[y_start:y_end, x_start:x_end] = expanded
    return result


def add_rect_border(image, border_width=10, color=(0, 0, 0)):
    """
    Рамка через слайсинг.
    """
    height, width, _ = image.shape
    bordered = image.copy()
    bordered[0:border_width, :] = color
    bordered[height - border_width:height, :] = color
    bordered[:, 0:border_width] = color
    bordered[:, width - border_width:width] = color
    return bordered


def add_shaped_border(image, border_width=10, color=(0, 0, 255), shape_type='wave'):
    """
    Фигурная рамка с использованием масок NumPy.
    """
    height, width, _ = image.shape
    bordered = image.copy()

    # Сетки координат
    Y, X = np.ogrid[:height, :width]

    if shape_type == 'wave':
        # Вычисляем оффсеты для всех x сразу
        offset_top = (border_width * (1 + np.sin(X / 10) / 2)).astype(int)
        offset_left = (border_width * (1 + np.sin(Y / 10) / 2)).astype(int)

        # Создаем булевы маски
        mask_top = Y < offset_top
        mask_bottom = Y > (height - offset_top)
        mask_left = X < offset_left
        mask_right = X > (width - offset_left)

        # Объединяем маски
        full_mask = mask_top | mask_bottom | mask_left | mask_right

        # Применяем цвет по маске
        bordered[full_mask] = color

    elif shape_type == 'dots':
        dot_size = 8
        dot_spacing = 12

        # Маски для полос, где могут быть точки
        # Используем остаток от деления для паттерна
        dots_x = (X % dot_spacing) < dot_size
        dots_y = (Y % dot_spacing) < dot_size

        # Зоны краев
        border_zone_y = (Y < dot_size) | (Y > height - dot_size)
        border_zone_x = (X < dot_size) | (X > width - dot_size)

        # Пересечение паттерна и зон
        mask_dots = (border_zone_y & dots_x) | (border_zone_x & dots_y)

        bordered[mask_dots] = color

    return bordered


def add_lens_flare(image, glare_path='glare.jpg', opacity=0.6):
    """
    Блик через наложение текстуры (без циклов).
    """
    glare = cv2.imread(glare_path)
    if glare is None:
        print(f"Ошибка: Файл блика {glare_path} не найден.")
        return image

    # Ресайзим блик под размер изображения с помощью нашей функции
    glare_resized = resize_image(glare, image.shape[1], image.shape[0])

    # Наложение (режим Screen или взвешенная сумма)
    # Формула: Result = Image + (Glare * Opacity), с клиппингом
    flared = image.astype(float) + (glare_resized.astype(float) * opacity)

    return np.clip(flared, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, texture_path='watercolor_texture.jpg', opacity=0.3):
    """
    Акварелизация через наложение текстуры.
    """
    texture = cv2.imread(texture_path)
    if texture is None:
        print(f"Ошибка: Текстура {texture_path} не найдена.")
        return image

    # Ресайзим текстуру
    texture_resized = resize_image(texture, image.shape[1], image.shape[0])

    # Взвешенная сумма
    textured = (image.astype(float) * (1 - opacity) + texture_resized.astype(float) * opacity)
    return textured.astype(np.uint8)


def main():
    global image_for_selection, ref_point

    parser = argparse.ArgumentParser(description="Применение фильтров к изображению")
    parser.add_argument("image_path", type=str, help="Путь к изображению")
    parser.add_argument("filter_type", type=str,
                        help="Тип фильтра: resize, sepia, vignette, pixelate, rect_border, shaped_border, lens_flare, watercolor")
    parser.add_argument("--params", nargs="*", default=[], help="Параметры фильтра")

    args = parser.parse_args()
    image = cv2.imread(args.image_path)

    if image is None:
        print("Изображение не найдено")
        return

    filter_type = args.filter_type
    params = args.params
    result = image.copy()

    if filter_type == "resize":  # пример: py lab1_filters.py input_image.jpg resize --params 800 600
        if len(params) >= 2:
            width, height = int(params[0]), int(params[1])
            result = resize_image(image, width, height)
        else:
            print("Нужны параметры ширины и высоты")
            return

    elif filter_type == "sepia":  # пример: py lab1_filters.py input_image.jpg sepia
        result = apply_sepia(image)

    elif filter_type == "vignette":  # пример: py lab1_filters.py input_image.jpg vignette --params 1
        strength = float(params[0]) if params else 0.5
        result = apply_vignette(image, strength)

    elif filter_type == "pixelate":  # пример: py lab1_filters.py input_image.jpg pixelate
        pixel_size = int(params[0]) if params else 10
        image_for_selection = image.copy()
        cv2.namedWindow("Select Region")
        cv2.setMouseCallback("Select Region", mouse_callback)

        print("Выделите область мышью и нажмите 'Enter' или 'Space' для применения фильтра. Нажмите 'c' для сброса.")

        while True:
            # Отображаем изображение (оно обновляется в колбэке)
            if not cropping:
                cv2.imshow("Select Region", image_for_selection)

            key = cv2.waitKey(1) & 0xFF

            # Enter (13) или Space (32) - применить
            if key == 13 or key == 32:
                if len(ref_point) == 2:
                    x1, y1 = ref_point[0]
                    x2, y2 = ref_point[1]
                    result = pixelate_region(image, x1, y1, x2, y2, pixel_size)
                    cv2.destroyWindow("Select Region")
                    break
            # 'c' - сбросить выделение
            elif key == ord("c"):
                image_for_selection = image.copy()
                ref_point = []

    elif filter_type == "rect_border":  # пример: py lab1_filters.py input_image.jpg rect_border --params 20 0 0 255
        # где: ширина=20, цвет=(0, 0, 255) - красный
        width = int(params[0]) if params else 10
        color = tuple(map(int, params[1:4])) if len(params) >= 4 else (0, 0, 255)
        result = add_rect_border(image, width, color)

    elif filter_type == "shaped_border":  # пример: py lab1_filters.py input_image.jpg shaped_border --params 20 0 0 255 wave
        # где: ширина=20, цвет=(0, 0, 255) - красный, wave - форма(wave/dots)
        width = int(params[0]) if params else 10
        color = tuple(map(int, params[1:4])) if len(params) >= 4 else (0, 0, 255)
        shape = params[4] if len(params) >= 5 else 'wave'
        result = add_shaped_border(image, width, color, shape)

    elif filter_type == "lens_flare":  # пример: py lab1_filters.py input_image.jpg lens_flare
        glare_file = 'glare.jpg'
        result = add_lens_flare(image, glare_file)

    elif filter_type == "watercolor":  # пример: py lab1_filters.py input_image.jpg watercolor --params 0.4
        texture_path = 'watercolor_texture.jpg'
        opacity = float(params[0]) if params else 0.3
        result = add_watercolor_texture(image, texture_path, opacity)

    else:
        print("Неизвестный фильтр")
        return

    cv2.imshow("Original", image)
    cv2.imshow("Filtered", result)
    print("Нажмите любую клавишу для выхода и сохранения...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_" + filter_type + ".jpg", result)
    print("Результат сохранен как output_" + filter_type + ".jpg")


if __name__ == "__main__":
    main()