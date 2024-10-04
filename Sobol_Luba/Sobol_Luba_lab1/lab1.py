import cv2 as cv
import argparse
import sys
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')
    parser.add_argument('-m', '--mode', 
                        help='Mode (image, grey_color,resize_image,sepia_filter,vignette_filter,pixelated)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-w', '--width',
                        help='Size (width)',
                        type=str,
                        default='200',
                        dest='width',)
    parser.add_argument('-hg', '--height',
                        help='Size (height)',                   
                        type=str,
                        default='200',
                        dest='height',)
    parser.add_argument('-px', '--pixel_size',
                        help='size pf pixel to pixelated',                   
                        type=str,
                        default='5',
                        dest='pixel_size',)

    args = parser.parse_args()

    return args


def highgui_samples(in_filename, out_filename):
    img = cv.imread(in_filename)
    cv.imwrite(out_filename, img)
    
    cv.imshow('Init image', img)
    cv.waitKey()
    
    cv.destroyAllWindows()

def grey_color(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    image = cv.imread(image_path)
    #height, width, _ = image.shape
    #grey_image = np.zeros((height, width, 3), dtype=np.uint8)

    # for i in range(height):
    #     for j in range(width):
    #         pixel = image[i, j]

    #         blue_channel = pixel[0]  # Синий канал
    #         green_channel = pixel[1]  # Зелёный канал
    #         red_channel = pixel[2]  # Красный канал

    #         grey_value = int(blue_channel * 0.114 + green_channel * 0.587 + red_channel * 0.299)
    #         grey_image[i, j] = [grey_value, grey_value, grey_value]
    blue_channel = image[:, :, 0] 
    green_channel = image[:, :, 1] 
    red_channel = image[:, :, 2] 
    
    grey_values = (blue_channel * 0.114 + green_channel * 0.587 + red_channel * 0.299).astype(np.uint8) 
    
    grey_image = np.stack((grey_values, grey_values, grey_values), axis=-1)
    col=np.vstack([image, grey_image])
    cv.imshow('Gray image',col)
    cv.waitKey()
    return grey_image

def resize_image(image_path, new_width, new_height):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    image = cv.imread(image_path)
    height, width, _ = image.shape
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            orig_x = int(j * width / new_width)
            orig_y = int(i * height / new_height)

            resized_image[i, j] = image[orig_y, orig_x]
    cv.imshow('resized_image',resized_image)
    cv.waitKey()
    return resized_image

def sepia_filter(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    image = cv.imread(image_path)
    # height, width, _ = image.shape
    # sepia_image = np.zeros((height, width, 3), dtype=np.uint8)

    # for i in range(height):
    #     for j in range(width):
    #         pixel = image[i, j]

    #         blue_channel = pixel[0]  # Синий канал
    #         green_channel = pixel[1]  # Зелёный канал
    #         red_channel = pixel[2]  # Красный канал

    #         tr = int(0.393 * red_channel + 0.769 * green_channel + 0.189 * blue_channel)
    #         tg = int(0.349 * red_channel + 0.686 * green_channel + 0.168 * blue_channel)
    #         tb = int(0.272 * red_channel + 0.534 * green_channel + 0.131 * blue_channel)

    #         # Ограничиваем значения до 255
    #         sepia_image[i, j] = [min(tb, 255), min(tg, 255), min(tr, 255)]
    blue_channel = image[:, :, 0] 
    green_channel = image[:, :, 1] 
    red_channel = image[:, :, 2] 
    
    sepia_blue = np.clip(0.272 * red_channel + 0.534 * green_channel + 0.131 * blue_channel, 0, 255).astype(np.uint8) 
    sepia_green = np.clip(0.349 * red_channel + 0.686 * green_channel + 0.168 * blue_channel, 0, 255).astype(np.uint8) 
    sepia_red = np.clip(0.393 * red_channel + 0.769 * green_channel + 0.189 * blue_channel, 0, 255).astype(np.uint8) 
     
    sepia_image = np.stack((sepia_blue, sepia_green, sepia_red), axis=-1)
    col=np.vstack([image, sepia_image])
    cv.imshow('sepia_image',col)
    cv.waitKey()
    return sepia_image

def vignette_filter(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    image = cv.imread(image_path)
    height, width, channels = image.shape
    # center_x, center_y = width // 2, height // 2
    # max_dist = np.sqrt(center_x**2 + center_y**2)

    # vignette_image = np.zeros((height, width, 3), dtype=np.uint8)

    # for i in range(height):
    #     for j in range(width):
    #         pixel = image[i, j]

    #         # Расстояние от центра
    #         dx = (j - center_x)**2
    #         dy = (i - center_y)**2
    #         distance = np.sqrt(dx + dy)

    #         # Виньетирование, чем дальше от центра, тем темнее пиксель
    #         factor = (max_dist - distance) / max_dist
    #         factor = max(factor, 0.1)  # Не даем фактору стать слишком маленьким

    #         vignette_image[i, j] = [int(channel * factor) for channel in pixel]
    center_x = width // 2 
    center_y = height // 2 
 
    max_dist = np.sqrt(center_x**2 + center_y**2) 

    x = np.arange(width) 
    y = np.arange(height) 
    xx, yy = np.meshgrid(x, y) 
    distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2) 
    
    factor = (max_dist - distances) / max_dist 
    factor = np.clip(factor, 0.1, 1.0)  # Не даем фактору стать слишком маленьким 
    
    vignette_image = np.zeros_like(image) 
    for channel in range(channels): 
        vignette_image[:, :, channel] = np.clip(image[:, :, channel] * factor, 0, 255).astype(np.uint8)
    col=np.vstack([image, vignette_image])
    cv.imshow('vignette_image',col)
    cv.waitKey()
    return vignette_image

def pixelate_area(image_path, pixel_size):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    image = cv.imread(image_path)
    height, width, _ = image.shape
    pixelated_image = np.copy(image)

    x1, y1 = (width // 4 ,height // 4) 
    x2, y2 = (3 * width // 4, 3 * height // 4)

    for i in range(y1, y2, pixel_size):
        for j in range(x1, x2, pixel_size):
            # Определяем усреднённый цвет для блока
            block = image[i:i+pixel_size, j:j+pixel_size]
            avg_color = block.mean(axis=(0, 1)).astype(int)

            # Закрашиваем блок усреднённым цветом
            pixelated_image[i:i+pixel_size, j:j+pixel_size] = avg_color
    col=np.vstack([image, pixelated_image])
    cv.imshow('vignette_image',col)
    cv.waitKey()
    return pixelated_image

def main():
    args = cli_argument_parser()
    
    if args.mode == 'image':
        highgui_samples(args.image_path, args.out_image_path)
    elif args.mode == 'grey_color':
        grey_color(args.image_path)
    elif args.mode == 'resize_image':
        resize_image(args.image_path,int(args.width), int(args.height))
    elif args.mode == 'sepia_filter':
        sepia_filter(args.image_path)
    elif args.mode == 'vignette_filter':
        vignette_filter(args.image_path)
    elif args.mode == 'pixelated':
        pixelate_area(args.image_path,int(args.pixel_size))
    else:
        raise ValueError('Unsupported mode')


if __name__ == '__main__':
    sys.exit(main() or 0)

