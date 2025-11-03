import cv2 as cv
import argparse
import sys
import numpy as np
import math
import random

def argument_parser():
    """Парсер командной строки для получения параметров"""
    parser = argparse.ArgumentParser(prog='lab1 - image processing',
                                     description="This laboratory work is devoted to basic operations on images.")

    # Основные параметры
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
                        help='Mode (resize, sepia, vig, pixel, border, fancy_border, lens_flare, watercolor)',
                        type=str,
                        required=True,
                        dest='mode')

    # Параметры для различных фильтров
    parser.add_argument('-c', '--coef',
                        help='Input coefficient for resolution change',
                        type=float,
                        dest='coef')

    parser.add_argument('-r', '--radius',
                        help='Input radius for vignette effect',
                        type=float,
                        dest='radius')

    parser.add_argument('-x', '--x_coord',
                        help='X coordinate for pixelation area',
                        type=int,
                        dest='x')

    parser.add_argument('-y', '--y_coord',
                        help='Y coordinate for pixelation area',
                        type=int,
                        dest='y')

    parser.add_argument('-w', '--width',
                        help='Width for pixelation area',
                        type=int,
                        dest='width')

    parser.add_argument('-ht', '--height',
                        help='Height for pixelation area',
                        type=int,
                        dest='height')

    parser.add_argument('-b', '--block',
                        help='Input block size for pixelation effect',
                        type=int,
                        dest='block')

    parser.add_argument('-bw', '--border_width',
                        help='Border width for border effect',
                        type=int,
                        dest='border_width')

    parser.add_argument('-bc', '--border_color',
                        help='Border color as B,G,R values',
                        type=str,
                        dest='border_color')

    parser.add_argument('-ft', '--frame_type',
                        help='Frame type for fancy border (wave, zigzag, triangle)',
                        type=str,
                        dest='frame_type')

    parser.add_argument('-fc', '--frame_color',
                        help='Frame color as B,G,R values',
                        type=str,
                        dest='frame_color')

    parser.add_argument('-fx', '--flare_x',
                        help='X coordinate for lens flare',
                        type=int,
                        dest='flare_x')

    parser.add_argument('-fy', '--flare_y',
                        help='Y coordinate for lens flare',
                        type=int,
                        dest='flare_y')

    return parser.parse_args()

def load_image(image_path):
    """Загрузка изображения"""
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError(f'Cannot load image from path: {image_path}')
    return image

def show_image_private(image, text):
    """Вспомогательная функция для отображения изображения"""
    window_name = text

    if image is not None:
        cv.imshow(window_name, image)
    else:
        raise ValueError('Empty image')

def show_images(original_image, result_image):
    """Отображение исходного и результирующего изображений"""
    show_image_private(original_image, 'original image')
    show_image_private(result_image,  'result_image')
    cv.waitKey(0)
    cv.destroyAllWindows()

def resize(original_image, scale):
    """Изменение разрешения изображения"""
    if scale <= 0:
        raise ValueError('Scale coefficient must be positive')
    
    height, width, number_channels = original_image.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    result_image = np.zeros((new_height, new_width, number_channels), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            result_image[i, j] = original_image[int(i/scale), int(j/scale)]
    
    return result_image

def sepia_filter(original_image):
    """Фильтр сепии"""
    height, width = original_image.shape[:2]
    result_image = np.zeros((height, width, 3), np.uint8)

    B = original_image[:, :, 0]
    G = original_image[:, :, 1]
    R = original_image[:, :, 2]

    result_image[:, :, 0] = np.clip(0.272 * R + 0.534 * G + 0.131 * B, 0, 255)
    result_image[:, :, 1] = np.clip(0.349 * R + 0.686 * G + 0.168 * B, 0, 255)
    result_image[:, :, 2] = np.clip(0.393 * R + 0.769 * G + 0.189 * B, 0, 255)

    return result_image

def vignette_img(original_image, radius):
    """Эффект виньетки"""
    rows, cols = original_image.shape[:2]
    
    # Генерация маски виньетки с использованием Гауссова ядра
    X_resultant_kernel = cv.getGaussianKernel(cols, radius)
    Y_resultant_kernel = cv.getGaussianKernel(rows, radius)
    
    # Создание матрицы ядра
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    
    # Создание маски и нормализация
    mask = resultant_kernel / resultant_kernel.max()
    
    result_image = np.copy(original_image)
    
    # Применение маски к каждому каналу входного изображения
    for i in range(3):
        result_image[:,:,i] = result_image[:,:,i] * mask

    return result_image.astype(np.uint8)

def pix_filter(original_image, x, y, width, height, block_size):
    """Пикселизация заданной области"""
    if original_image is None:
        raise ValueError('Empty image')
    if width <= 0 or height <= 0:
        raise ValueError('Invalid area dimensions')
    if block_size <= 0:
        raise ValueError('Block size must be positive')
    
    result_image = np.copy(original_image)
    img_height, img_width = original_image.shape[:2]
    
    # Проверка границ
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)

    for i in range(y, y + height, block_size):
        for j in range(x, x + width, block_size):
            # Извлечение блока
            block_end_i = min(i + block_size, y + height)
            block_end_j = min(j + block_size, x + width)
            block = result_image[i:block_end_i, j:block_end_j]
            
            if block.size == 0:
                continue
                
            # Вычисление среднего цвета блока
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            result_image[i:block_end_i, j:block_end_j] = avg_color
            
    return result_image

def add_border(original_image, border_width, border_color):
    """Добавление прямоугольной рамки"""
    if border_width <= 0:
        raise ValueError('Border width must be positive')
    
    height, width = original_image.shape[:2]
    
    # Создание изображения с рамкой
    result_image = np.copy(original_image)
    
    # Закрашивание рамки
    result_image[0:border_width, :] = border_color  # Верхняя рамка
    result_image[height-border_width:height, :] = border_color  # Нижняя рамка
    result_image[:, 0:border_width] = border_color  # Левая рамка
    result_image[:, width-border_width:width] = border_color  # Правая рамка
    
    return result_image

def add_fancy_border(original_image, frame_type, frame_color):
    """Добавление фигурной рамки"""
    height, width = original_image.shape[:2]
    border_width = min(height, width) // 10  # Ширина рамки - 10% от меньшей стороны
    
    result_image = np.copy(original_image)
    
    if frame_type == 'wave':
        # Волнистая рамка
        for i in range(border_width):
            for j in range(width):
                if i < border_width * (0.5 + 0.5 * math.sin(j * 2 * math.pi / 50)):
                    result_image[i, j] = frame_color
                    result_image[height-1-i, j] = frame_color
                    
        for i in range(height):
            for j in range(border_width):
                if j < border_width * (0.5 + 0.5 * math.sin(i * 2 * math.pi / 50)):
                    result_image[i, j] = frame_color
                    result_image[i, width-1-j] = frame_color
                    
    elif frame_type == 'zigzag':
        # Зигзагообразная рамка
        for i in range(border_width):
            for j in range(width):
                if i < border_width * (0.5 + 0.5 * abs((j % 20) - 10) / 10):
                    result_image[i, j] = frame_color
                    result_image[height-1-i, j] = frame_color
                    
        for i in range(height):
            for j in range(border_width):
                if j < border_width * (0.5 + 0.5 * abs((i % 20) - 10) / 10):
                    result_image[i, j] = frame_color
                    result_image[i, width-1-j] = frame_color
                    
    elif frame_type == 'triangle':
        # Треугольная рамка
        for i in range(border_width):
            for j in range(width):
                if i < border_width * (1 - abs(j - width/2) / (width/2)):
                    result_image[i, j] = frame_color
                    result_image[height-1-i, j] = frame_color
                    
        for i in range(height):
            for j in range(border_width):
                if j < border_width * (1 - abs(i - height/2) / (height/2)):
                    result_image[i, j] = frame_color
                    result_image[i, width-1-j] = frame_color
    
    return result_image

def add_lens_flare(original_image, flare_x, flare_y):
    """Эффект бликов объектива"""
    height, width = original_image.shape[:2]
    result_image = np.copy(original_image).astype(np.float32)
    
    # Создание нескольких бликов разного размера и интенсивности
    flares = [
        (flare_x, flare_y, 50, 1.0),  # Основной блик
        (flare_x - 30, flare_y - 30, 30, 0.7),
        (flare_x + 40, flare_y + 20, 25, 0.5),
        (flare_x - 50, flare_y + 30, 20, 0.4)
    ]
    
    for fx, fy, size, intensity in flares:
        for i in range(max(0, fy-size), min(height, fy+size)):
            for j in range(max(0, fx-size), min(width, fx+size)):
                distance = math.sqrt((j - fx)**2 + (i - fy)**2)
                if distance < size:
                    # Уменьшение интенсивности к краям блика
                    flare_strength = intensity * (1 - distance/size)
                    
                    # Добавление блика (увеличиваем яркость)
                    for channel in range(3):
                        result_image[i, j, channel] = min(
                            255, 
                            result_image[i, j, channel] + flare_strength * 100
                        )
    
    return result_image.astype(np.uint8)

def add_watercolor_texture(original_image):
    """Наложение текстуры акварельной бумаги"""
    height, width = original_image.shape[:2]
    result_image = np.copy(original_image).astype(np.float32)
    
    # Создание текстуры акварельной бумаги (зернистость)
    texture = np.random.normal(0, 15, (height, width, 3))
    
    # Применение текстуры
    result_image = result_image + texture
    
    # Легкое размытие для имитации акварели
    temp_image = result_image.copy()
    for i in range(1, height-1):
        for j in range(1, width-1):
            for channel in range(3):
                # Простое размытие
                result_image[i, j, channel] = np.mean(temp_image[i-1:i+2, j-1:j+2, channel])
    
    return np.clip(result_image, 0, 255).astype(np.uint8)

def parse_color(color_str):
    """Парсинг строки цвета в формате B,G,R"""
    if color_str is None:
        return [0, 0, 0]  # Черный по умолчанию
    
    try:
        b, g, r = map(int, color_str.split(','))
        return [b, g, r]
    except:
        raise ValueError('Color must be in format B,G,R')

def main():
    args = argument_parser()
    original_image = load_image(args.image_path)
    result_image = None
    
    if args.mode == 'resize':
        if args.coef is None:
            raise ValueError('Coefficient is required for resize mode')
        result_image = resize(original_image, args.coef)
        
    elif args.mode == 'sepia':
        result_image = sepia_filter(original_image)
        
    elif args.mode == 'vig':
        if args.radius is None:
            raise ValueError('Radius is required for vignette mode')
        result_image = vignette_img(original_image, args.radius)

    elif args.mode == 'pixel':
        if args.x is None or args.y is None or args.width is None or args.height is None or args.block is None:
            raise ValueError('All coordinates and block size are required for pixel mode')
        result_image = pix_filter(original_image, args.x, args.y, args.width, args.height, args.block)

    elif args.mode == 'border':
        if args.border_width is None:
            raise ValueError('Border width is required for border mode')
        border_color = parse_color(args.border_color)
        result_image = add_border(original_image, args.border_width, border_color)

    elif args.mode == 'fancy_border':
        if args.frame_type is None:
            raise ValueError('Frame type is required for fancy_border mode')
        frame_color = parse_color(args.frame_color)
        result_image = add_fancy_border(original_image, args.frame_type, frame_color)

    elif args.mode == 'lens_flare':
        if args.flare_x is None or args.flare_y is None:
            raise ValueError('Flare coordinates are required for lens_flare mode')
        result_image = add_lens_flare(original_image, args.flare_x, args.flare_y)

    elif args.mode == 'watercolor':
        result_image = add_watercolor_texture(original_image)

    else:
        raise ValueError(f'Unknown mode: {args.mode}')

    show_images(original_image, result_image)
    
    # Сохранение результата
    cv.imwrite(args.out_image_path, result_image)

if __name__ == '__main__':
    sys.exit(main() or 0)
