import cv2 as cv
import argparse
import sys
import numpy as np
import random


def cli_argument_parser():
    """Парсер командной строки для получения параметров"""
    parser = argparse.ArgumentParser(description="Image processing tool")

    # Параметры командной строки
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        required=True,
                        dest='image_path')

    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')

    parser.add_argument('-m', '--mode',
                        help='Mode (gray, res, sepia, vig, pixel, pixel2)',
                        type=str,
                        default='image',
                        dest='mode')

    parser.add_argument('-c', '--coef',
                        help='Input coefficient for resolution change',
                        type=float,
                        dest='coef')

    parser.add_argument('-r', '--radius',
                        help='Input radius for vignette effect',
                        type=float,
                        dest='radius')

    parser.add_argument('-b', '--block',
                        help='Input block size for pixelation effect',
                        type=int,
                        dest='block')

    return parser.parse_args()


def read_img(image_path):
    """Чтение изображения"""
    if image_path is None:
        raise ValueError('Empty path to the image')
    return cv.imread(image_path)


def show_img(text, image, image2):
    """Отображение оригинального и обработанного изображения"""
    if image is None:
        raise ValueError('Empty path to the image')

    cv.imshow('Original Image', image)
    cv.imshow(text, image2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def gray_img(src_image):
    """Преобразование изображения в оттенки серого"""
    gray_image = 0.299 * src_image[:, :, 0] + 0.587 * src_image[:, :, 1] + 0.114 * src_image[:, :, 2]
    return gray_image.astype(np.uint8)


def resolution_img(src_image, coef):
    """Изменение разрешения изображения"""
    height, width = src_image.shape[:2]

    new_height, new_width = int(height * coef), int(width * coef)

    x_ind = np.floor(np.arange(new_width)/coef).astype(int)
    y_ind = np.floor(np.arange(new_height)/coef).astype(int)
    result_image = src_image[y_ind[:,None], x_ind]
    """for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * src_image.shape[1] / new_width)
            src_y = int(y * src_image.shape[0] / new_height)
            resolution_image[y, x] = src_image[src_y, src_x]"""
    
    return result_image


def sepia_img(src_image):
    """Применение сепии к изображению"""
    height, width = src_image.shape[:2]
    sepia_image = np.zeros((height, width, 3), np.uint8)

    gray = 0.299 * src_image[:, :, 2] + 0.587 * src_image[:, :, 1] + 0.114 * src_image[:, :, 0]
    sepia_image[:, :, 0] = np.clip(gray - 30, 0, 255)
    sepia_image[:, :, 1] = np.clip(gray + 15, 0, 255)
    sepia_image[:, :, 2] = np.clip(gray + 40, 0, 255)

    return sepia_image


def vignette_img(img, radius):
    rows, cols = img.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, radius)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, radius)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    processed_img = img.copy()

    for i in range(3):  # Apply to each channel
        processed_img[:,:,i] = processed_img[:,:,i] * mask

    return processed_img


def area(image):
    """Выбор области для пикселизации"""
    new_x, new_y, new_width, new_height = 0, 0, 0, 0

    def mouse_click(event, x, y, flags, param):
        nonlocal new_x, new_y, new_width, new_height
        if event == cv.EVENT_LBUTTONDOWN:
            new_x, new_y = x, y
        elif event == cv.EVENT_LBUTTONUP:
            new_width = x - new_x
            new_height = y - new_y

    cv.imshow('Area', image)
    cv.setMouseCallback('Area', mouse_click)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (new_x, new_y, new_width, new_height)


def pixel_img(src_image, block_size, x, y, width, height):
    """Пикселизация изображения"""

    pixel_img = np.zeros_like(src_image)
    np.copyto(pixel_img, src_image)

    roi = pixel_img[y:y + height, x:x + width]
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = roi[i:i + block_size, j:j + block_size]
            color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            roi[i:i + block_size, j:j + block_size] = color

    pixel_img[y:y + height, x:x + width] = roi
    return pixel_img


def main():
    """Основная функция программы"""
    # Получаем аргументы из командной строки
    args = cli_argument_parser()

    # Загружаем изображение
    src_image = read_img(args.image_path)

    # Выбираем режим обработки изображения
    if args.mode == 'gray':
        new_image = gray_img(src_image)
        text = 'Gray image'
    elif args.mode == 'res':
        new_image = resolution_img(src_image, args.coef)
        text = 'Resolution image'
    elif args.mode == 'sepia':
        new_image = sepia_img(src_image)
        text = 'Sepia image'
    elif args.mode == 'vig':
        new_image = vignette_img(src_image, args.radius)
        text = 'Vignette image'
    elif args.mode == 'pixel':
        x, y, width, height = area(src_image)
        new_image = pixel_img(src_image, args.block, x, y, width, height)
        text = 'Pixel image'
    else:
        raise ValueError('Unsupported mode')

    # Отображаем результат
    show_img(text, src_image, new_image)


if __name__ == '__main__':
    sys.exit(main() or 0)
