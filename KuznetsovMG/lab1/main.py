import argparse
import sys
import cv2 as cv
import numpy as np

# Функция для разбора аргументов командной строки
def get_arguments():
    parser = argparse.ArgumentParser(description="Программа для обработки изображений")
    parser.add_argument(
        '-m', '--mode',
        type=str,
        default='image',
        dest='mode',
        help="Режим обработки ('grayImage', 'resolImage', 'sepiaImage', 'vignetteImage', 'pixelImage')"
    )
    parser.add_argument(
        '-i', '--image',
        type=str,
        required=True,
        dest='input_path',
        help="Путь к входному изображению"
    )
    parser.add_argument(
        '-p', '--params',
        type=str,
        dest='params',
        help="Параметры для выбранного режима, разделённые запятыми"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        dest='output_path',
        default="",
        help="Путь для сохранения обработанного изображения (опционально)"
    )
    return parser.parse_args()

# Функция для преобразования строки параметров в список чисел
def split_params(param_str):
    if param_str is None:
        return [], 0
    values = [float(val) for val in param_str.split(",")]
    return values, len(values)

# Функция для загрузки изображения
def load_image(filepath):
    img = cv.imread(filepath)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {filepath}")
    return img

# Функция для показа оригинального и обработанного изображений
def show_images(original, processed, orig_title="Оригинал", proc_title="Обработанное"):
    cv.namedWindow(orig_title, cv.WINDOW_NORMAL)
    cv.namedWindow(proc_title, cv.WINDOW_NORMAL)
    cv.imshow(orig_title, original)
    cv.imshow(proc_title, processed)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Преобразование изображения в оттенки серого по формуле яркости
def convert_to_gray(image):
    gray = (0.299 * image[:, :, 2] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 0]).astype(np.uint8)
    return gray

# Функция изменения размера изображения с использованием ближайшего соседа
def scale_image(image, new_width, new_height):
    orig_h, orig_w = image.shape[:2]
    resized = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            src_x = int(j * orig_w / new_width)
            src_y = int(i * orig_h / new_height)
            resized[i, j] = image[src_y, src_x]
    return resized

# Применение сепия-эффекта к изображению
def sepia_transform(image):
    sepia = np.zeros_like(image, dtype=np.float32)
    sepia[:, :, 0] = 0.393 * image[:, :, 2] + 0.769 * image[:, :, 1] + 0.189 * image[:, :, 0]  # Синий канал
    sepia[:, :, 1] = 0.349 * image[:, :, 2] + 0.686 * image[:, :, 1] + 0.168 * image[:, :, 0]  # Зеленый канал
    sepia[:, :, 2] = 0.272 * image[:, :, 2] + 0.534 * image[:, :, 1] + 0.131 * image[:, :, 0]  # Красный канал
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

# Применение виньеточного эффекта с заданным радиусом
def add_vignette(image, radius):
    h, w = image.shape[:2]
    center_x, center_y = h // 2, w // 2
    scale_a, scale_b = 1, 1
    if h > w:
        scale_a = h / w
    elif w > h:
        scale_b = w / h
    y_idx, x_idx = np.indices((h, w))
    dist = np.sqrt(((y_idx - center_x) / scale_a) ** 2 + ((x_idx - center_y) / scale_b) ** 2)
    mask = 1 - np.minimum(1, dist / radius)
    vignette_img = image.copy()
    vignette_img[:, :, 0] = vignette_img[:, :, 0] * mask
    vignette_img[:, :, 1] = vignette_img[:, :, 1] * mask
    vignette_img[:, :, 2] = vignette_img[:, :, 2] * mask
    return vignette_img.astype(np.uint8)

# Глобальные переменные для определения области пикселизации
rect_start_x, rect_start_y, rect_end_x, rect_end_y = -1, -1, -1, -1
is_drawing = False

# Функция для пикселизации выделенной области
def pixelate(image, block_size):
    global rect_start_x, rect_start_y, rect_end_x, rect_end_y
    if rect_start_x == rect_start_y == rect_end_x == rect_end_y == -1:
        rect_start_x, rect_start_y = 0, 0
        rect_end_x, rect_end_y = image.shape[1], image.shape[0]
    region = image[rect_start_y:rect_end_y, rect_start_x:rect_end_x]
    reg_h, reg_w = region.shape[:2]
    num_blocks_y = reg_h // block_size
    num_blocks_x = reg_w // block_size
    for i in range(num_blocks_y):
        start_y = i * block_size
        end_y = min(start_y + block_size, reg_h)
        for j in range(num_blocks_x):
            start_x = j * block_size
            end_x = min(start_x + block_size, reg_w)
            block_avg = region[start_y:end_y, start_x:end_x].mean(axis=(0, 1)).astype(int)
            region[start_y:end_y, start_x:end_x] = block_avg
    image[rect_start_y:rect_end_y, rect_start_x:rect_end_x] = region
    return image

# Обработчик событий мыши для выбора области
def mouse_callback(event, x, y, flags, param):
    global is_drawing, rect_start_x, rect_start_y, rect_end_x, rect_end_y
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True
        rect_start_x, rect_start_y = x, y
    elif event == cv.EVENT_MOUSEMOVE and is_drawing:
        rect_end_x, rect_end_y = x, y
    elif event == cv.EVENT_LBUTTONUP:
        is_drawing = False
        rect_end_x, rect_end_y = x, y

# Функция для выбора области пикселизации с помощью мыши
def pixelate_area(image, block_size):
    global is_drawing, rect_start_x, rect_start_y, rect_end_x, rect_end_y
    window = 'Select area for pixelation'
    cv.namedWindow(window)
    cv.setMouseCallback(window, mouse_callback)
    while True:
        temp_img = image.copy()
        if rect_start_x != -1 and rect_start_y != -1 and rect_end_x != -1 and rect_end_y != -1:
            cv.rectangle(temp_img, (rect_start_x, rect_start_y), (rect_end_x, rect_end_y), (0, 255, 0), 2)
        cv.imshow(window, temp_img)
        if cv.waitKey(1) == 13:  # клавиша Enter
            cv.destroyWindow(window)
            break
    return pixelate(image, block_size)

def main():
    args = get_arguments()
    img = load_image(args.input_path)
    params_list, num_params = split_params(args.params)
    
    mode = args.mode
    if mode == 'grayImage':
        result = convert_to_gray(img)
    elif mode == 'resolImage':
        if num_params != 2:
            raise ValueError("Для изменения размера нужны два параметра: новая ширина и новая высота.")
        result = scale_image(img, int(params_list[0]), int(params_list[1]))
    elif mode == 'sepiaImage':
        result = sepia_transform(img)
    elif mode == 'vignetteImage':
        if num_params != 1:
            raise ValueError("Для виньетки требуется один параметр: радиус эффекта.")
        result = add_vignette(img, params_list[0])
    elif mode == 'pixelImage':
        if num_params != 1:
            raise ValueError("Для пикселизации требуется один параметр: размер блока.")
        result = pixelate_area(img, int(params_list[0]))
    else:
        print("Ошибка: неподдерживаемый режим обработки.")
        return 1

    if result is not None:
        show_images(img, result)
        if args.output_path:
            try:
                cv.imwrite(args.output_path, result)
                print(f"Изображение сохранено по адресу: {args.output_path}")
            except Exception as e:
                print(f"Ошибка при сохранении изображения: {e}")
    else:
        print("Обработка завершилась с ошибкой.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
