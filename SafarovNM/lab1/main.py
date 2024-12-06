import argparse
import sys
import cv2
import numpy as np

# Глобальные переменные для хранения координат прямоугольника
start_point = None
end_point = None
drawing = False

def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Image processing tool')

    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\', \'grey_color\', \'permissions\', \'sepia_filter\', \'vignette_filter\', \'pixelated\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--image',
                        help='Sets the path to the original image.',
                        type=str,
                        dest='image_path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Sets the name of the processed image.',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('-p', '--params',
                        help='Parameters for the mode (comma-separated).',
                        type=str,
                        dest='params')

    args = parser.parse_args()
    return args

def validate_args(args):
    if args.mode not in ['image', 'grey_color', 'permissions', 'sepia_filter', 'vignette_filter', 'pixelated']:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if args.image_path is None:
        raise ValueError("Image path is required")

    if args.mode in ['permissions', 'vignette_filter', 'pixelated']:
        if args.params is None:
            raise ValueError(f"Parameters are required for mode: {args.mode}")
        params = args.params.split(',')
        if args.mode == 'permissions' and len(params) != 2:
            raise ValueError("Two parameters (height, width) are required for 'permissions' mode")
        if args.mode == 'vignette_filter' and len(params) != 2:
            raise ValueError("Two parameters (radius, intensity) are required for 'vignette_filter' mode")
        if args.mode == 'pixelated' and len(params) != 1:
            raise ValueError("One parameter (block_size) is required for 'pixelated' mode")

def get_height_img(path):
    return load_image(path).shape[0]

def get_width_img(path):
    return load_image(path).shape[1]

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    return img

def show_image(img, type):
    if type == 'R':
        cv2.imshow('Result', img)
    else:
        cv2.imshow('Source', img)

    cv2.waitKey(0)

def save_and_show_images(src_img, processed_img, output_path):
    cv2.imwrite(output_path, processed_img)
    cv2.imshow('Original Image', src_img)
    cv2.imshow('Processed Image', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grey_color(img_path):
    if img_path is None:
        raise ValueError('Empty path to the image')

    src_img = load_image(img_path)

    grey_img = np.zeros((get_height_img(img_path), get_width_img(img_path)), dtype=float)
    grey_img[:, :] = src_img[:, :, 0] * 0.299 + src_img[:, :, 1] * 0.587 + src_img[:, :, 2] * 0.114

    return grey_img.astype(np.uint8)

def resolution_img(img_path, height, width):
    if img_path is None:
        raise ValueError('Empty path to the image')

    src_img = load_image(img_path)
    current_height, current_width = get_height_img(img_path), get_width_img(img_path)

    # Создаем массив для нового изображения
    resized_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Вычисляем коэффициенты масштабирования
    height_ratio = current_height / height
    width_ratio = current_width / width

    # Перебираем каждый пиксель нового изображения
    for i in range(height):
        for j in range(width):
            # Вычисляем координаты пикселя в исходном изображении
            src_i = int(i * height_ratio)
            src_j = int(j * width_ratio)

            # Копируем значение пикселя из исходного изображения в новое
            resized_img[i, j] = src_img[src_i, src_j]

    return resized_img

def sepia_filter(img_path):
    if img_path is None:
        raise ValueError('Empty path to the image')

    src_image = load_image(img_path)
    sepia_img = np.array(src_image, dtype=np.float64)  # Преобразуем в float для удобства вычислений

    # Применяем матрицу преобразования для сепии
    sepia_img = cv2.transform(sepia_img, np.matrix([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ]))

    # Ограничиваем значения пикселей до 255
    sepia_img[np.where(sepia_img > 255)] = 255

    # Преобразуем обратно в uint8
    sepia_img = np.array(sepia_img, dtype=np.uint8)

    return sepia_img

def vignette_filter(img_path, radius, intensity):
    if img_path is None:
        raise ValueError('Empty path to the image')

    src_image = load_image(img_path)
    height, width = get_height_img(img_path), get_width_img(img_path)

    # Создаем сетку координат
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Вычисляем центр изображения
    center_x = width / 2
    center_y = height / 2

    # Вычисляем расстояние от центра для каждого пикселя
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Нормализуем расстояние к диапазону [0, 1]
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    normalized_distance = distance / max_distance

    # Создаем эффект виньетки (обратная зависимость от расстояния)
    vignette_mask = 1 - normalized_distance ** 2  # Используем квадратичное затухание
    vignette_mask = np.clip(vignette_mask, 0, 1)  # Убираем отрицательные значения

    # Применяем эффект виньетки к каждому каналу изображения
    vignette_img = np.zeros_like(src_image, dtype=np.uint8)
    for i in range(3):  # Для каждого канала (R, G, B)
        vignette_img[:, :, i] = (src_image[:, :, i] * vignette_mask).astype(np.uint8)

    return vignette_img

def select_roi(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        pixelated_img = pixelate_roi(param.copy(), start_point, end_point, block_size)
        cv2.rectangle(param, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select ROI", param)
        cv2.imshow("Processed Image", pixelated_img)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image_copy = param.copy()
            cv2.rectangle(image_copy, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", image_copy)

def pixelate_roi(img, start_point, end_point, block_size):
    x1, y1 = start_point
    x2, y2 = end_point
    roi_img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)].copy()
    height, width = roi_img.shape[:2]

    # Создаем маску для пикселизации
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Вычисляем среднее значение пикселей в блоке
            block = roi_img[y:y+block_size, x:x+block_size]
            avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)

            # Заменяем все пиксели в блоке на среднее значение
            roi_img[y:y+block_size, x:x+block_size] = avg_color

    # Заменяем оригинальный ROI на пикселизованный
    img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = roi_img
    return img

def main():
    global block_size
    args = cli_argument_parser()
    validate_args(args)

    src_img = load_image(args.image_path)

    if args.mode == 'grey_color':
        processed_img = grey_color(args.image_path)
    elif args.mode == 'permissions':
        params = args.params.split(',')
        processed_img = resolution_img(args.image_path, int(params[0]), int(params[1]))
    elif args.mode == 'sepia_filter':
        processed_img = sepia_filter(args.image_path)
    elif args.mode == 'vignette_filter':
        params = args.params.split(',')
        processed_img = vignette_filter(args.image_path, float(params[0]), float(params[1]))
    elif args.mode == 'pixelated':
        params = args.params.split(',')
        block_size = int(params[0])
        cv2.namedWindow('Select ROI')
        cv2.setMouseCallback('Select ROI', select_roi, param=src_img.copy())
        cv2.imshow('Select ROI', src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        processed_img = src_img

    save_and_show_images(src_img, processed_img, args.output_image)

if __name__ == '__main__':
    sys.exit(main() or 0)