import argparse
import sys
import cv2 as cv
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='output.jpg',
                        dest='out_image_path')
    parser.add_argument('-m', '--mode',
                        help='Mode (image, grey_color, change_resolution, sepia_filter, vignette_filter, pixelated)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-w', '--width',
                        help='New width for resizing (only for resize_image mode)',
                        type=int,
                        default=200,
                        dest='width')
    parser.add_argument('-hg', '--height',
                        help='New height for resizing (only for resize_image mode)',
                        type=int,
                        default=200,
                        dest='height')
    parser.add_argument('-px', '--pixel_size',
                        help='Size of pixels for pixelation (only for pixelated mode)',
                        type=int,
                        default=5,
                        dest='pixel_size')

    args = parser.parse_args()
    return args

def image_mode(image_path, out_image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    cv.imwrite(out_image_path, image)
    print(f"Оригинальное изображение сохранено в '{out_image_path}'.")

def grey_color(image_path, out_image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    height, width, channels = image.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)
    
    # Проходим по каждому пикселю и вычисляем значение серого
    for i in range(height):
        for j in range(width):
            # Получаем значения R, G, B для текущего пикселя
            B, G, R = image[i, j]
            # Вычисляем значение серого
            gray_value = int(0.299 * R + 0.587 * G + 0.114 * B)
            # Устанавливаем значение серого в новом изображении
            gray_image[i, j] = gray_value
    
    # Преобразуем обратно в 3-канальный BGR
    gray_image_colored = cv.merge([gray_image, gray_image, gray_image])
    
    cv.imwrite(out_image_path, gray_image_colored)
    print(f"Грейскейл изображение сохранено в '{out_image_path}'.")

def change_resolution(image_path, new_width, new_height, out_image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Коэффициенты масштабирования по ширине и высоте
    x_ratio = original_width / new_width
    y_ratio = original_height / new_height

    for i in range(new_height):
        for j in range(new_width):
            # Определяем координаты соответствующего пикселя в исходном изображении
            orig_x = int(j * x_ratio)
            orig_y = int(i * y_ratio)

            # Копируем значение пикселя
            resized_image[i, j] = image[orig_y, orig_x]

    cv.imwrite(out_image_path, resized_image)
    print(f"Изменённое разрешение изображения сохранено в '{out_image_path}'.")

def sepia_filter(image_path, out_image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    height, width, channels = image.shape
    sepia_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            B, G, R = image[i, j]
            # Вычисляем новые значения RGB
            new_R = min(255, int(0.393 * R + 0.769 * G + 0.189 * B))
            new_G = min(255, int(0.349 * R + 0.686 * G + 0.168 * B))
            new_B = min(255, int(0.272 * R + 0.534 * G + 0.131 * B))
            sepia_image[i, j] = [new_B, new_G, new_R]
    
    cv.imwrite(out_image_path, sepia_image)
    print(f"Сепия изображение сохранено в '{out_image_path}'.")


def vignette_filter(image_path, out_image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    rows, cols = image.shape[:2]
    vignette_image = np.copy(image)
    
    # Генерация маски виньетки
    center_x, center_y = cols // 2, rows // 2
    for i in range(rows):
        for j in range(cols):
            # Расстояние от текущего пикселя до центра
            distance = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
            # Вычисляем коэффициент ослабления
            vignette_strength = max(0, 1 - distance / max(center_x, center_y))
            # Применяем маску к каждому цветному каналу
            for c in range(3):
                vignette_image[i, j, c] = int(vignette_image[i, j, c] * vignette_strength)
    
    cv.imwrite(out_image_path, vignette_image)
    print(f"Виньетированное изображение сохранено в '{out_image_path}'.")

def pixelate_image(image_path, pixel_size, out_image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    height, width = image.shape[:2]
    pixelated_image = np.zeros_like(image)

    # Пикселизация
    for i in range(0, height, pixel_size):
        for j in range(0, width, pixel_size):
            # Находим размеры блока
            block = image[i:i + pixel_size, j:j + pixel_size]
            # Вычисляем средний цвет блока
            avg_color = block.mean(axis=(0, 1))
            # Применяем средний цвет ко всем пикселям блока
            pixelated_image[i:i + pixel_size, j:j + pixel_size] = avg_color
    
    cv.imwrite(out_image_path, pixelated_image)
    print(f"Пикселизированное изображение сохранено в '{out_image_path}'.")


# Главная функция
def main():
    #python lab1.py -i input.jpg -o output.jpg -m vignette_filter
    args = cli_argument_parser()

    if args.mode == 'image':
        image_mode(args.image_path, args.out_image_path)
    elif args.mode == 'grey_color':
        grey_color(args.image_path, args.out_image_path)
    elif args.mode == 'change_resolution':
        change_resolution(args.image_path, int(args.width), int(args.height), args.out_image_path)
    elif args.mode == 'sepia_filter':
        sepia_filter(args.image_path, args.out_image_path)
    elif args.mode == 'vignette_filter':
        vignette_filter(args.image_path, args.out_image_path)
    elif args.mode == 'pixelated':
        pixelate_image(args.image_path, int(args.pixel_size), args.out_image_path)
    else:
        raise ValueError('Unsupported mode')

    print(f"Фильтр '{args.mode}' успешно применён. Изображение сохранено в '{args.out_image_path}'.")

if __name__ == '__main__':
    sys.exit(main() or 0)


