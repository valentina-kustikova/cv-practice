#Импортируем необходимые библиотеки
import cv2 as cv
import numpy as np
import argparse
import sys

#Пишем функцию, которая создаёт объект argparse.ArgumentParser для разбора аргументов командной строки
def cli_argument_parser():
    parser = argparse.ArgumentParser()
    
    #Определяются аргументы командной строки
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\', \'image_to_grayscale\', \'image_resize\', \'sepia\', \'vignette\', \'pixelation\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-v', '--value',
                        help='input value for filter',
                        type=float,
                        dest='value',
                        default='1')
    parser.add_argument('-p', '--pixel_size',
                        help='size pf pixel',                   
                        type=str,
                        dest='pixel_size',
                        default='5')
    
    args = parser.parse_args()
    return args

#Пишем функцию, которая занимается базовыми операциями с изображениями с помощью библиотеки OpenCV
def highgui_image_samples(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')

    #Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    #Отображение изображения
    cv.imshow('Init image', image)
    cv.waitKey(0)
    
    #Сохранение изображения
    cv.imwrite(output_image, image)

    #Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()


#Функция перевода изображения в оттенки серого
def image_to_grayscale(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    #Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape
    
    grey_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            R, G, B = image[i, j]
            grey_value = int(0.2989 * R + 0.5870 * G + 0.1140 * B) 
            grey_image[i, j] = grey_value
    
    #Отображение изображения
    cv.imshow('Init image', image)
    cv.imshow('Output image', grey_image)
    cv.waitKey(0)
    
    #Сохранение изображения
    cv.imwrite(output_image, grey_image)

    #Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()


#Функция изменения разрешения изображения
def image_resize(image_path, output_image, value):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    #Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape
    
    
    new_height = int(height*value)
    new_width = int(width*value)
    
    resize_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            new_x = int(j/value)
            new_y = int(i/value)
            
            resize_image[i, j] = image[new_y, new_x]

    #Отображение изображения
    cv.imshow('Init image', image)
    cv.imshow('Output image', resize_image)
    cv.waitKey(0)

    #Сохранение изображения
    cv.imwrite(output_image, resize_image)

    #Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()


#Функция применения фотоэффекта сепии к изображению.
def sepia(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    #Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    sepia_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            B, G, R = image[i, j]
            sepiaR = min(int(0.393 * R + 0.769 * G + 0.189 * B), 255)
            sepiaG = min(int(0.349 * R + 0.686 * G + 0.168 * B), 255)
            sepiaB = min(int(0.272 * R + 0.534 * G + 0.131 * B), 255)

            sepia_image[i, j] = [sepiaB, sepiaG, sepiaR]

    #Отображение изображения
    cv.imshow('Init image', image)
    cv.imshow('Output image', sepia_image)
    cv.waitKey(0)

    #Сохранение изображения
    cv.imwrite(output_image, sepia_image)

    #Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()
     
 
#Функция применения фотоэффекта виньетки к изображению.
def vignette(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    #Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    vignette_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    x = int(width / 2)
    y = int(height / 2)
    for i in range(height):
        for j in range(width):
            x_value = 1 - (abs(j - x) / x)
            y_value = 1 - (abs(i - y) / y)
            vignette_image[i, j] = image[i, j]  * x_value * y_value
    
    #Отображение изображения
    cv.imshow('Init image', image)
    cv.imshow('Output image', vignette_image)
    cv.waitKey(0)

    #Сохранение изображения
    cv.imwrite(output_image, vignette_image)

    #Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()
    
    
#Функция пикселизации заданной прямоугольной области изображения
def pixelation(image_path, output_image, pixel_size):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    #Загрузка изображения
    image = cv.imread(image_path)
    height, width, nchannels = image.shape

    pixelation_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    pixelation_image = image.copy()
    
    x1 = width // 4
    x2 = int(width * 0.5)
    y1 = height // 4
    y2 = int(height * 0.5)

    for i in range(x1, x2, pixel_size):
        for j in range(y1, y2, pixel_size):
            
            region = image[j:j + pixel_size, i:i + pixel_size]
            
            mean_color_bgr = np.zeros(nchannels, dtype=int)
            for k in range(nchannels):
                mean_color_bgr[k] = np.mean(region[:, :, k])
                   
            pixelation_image[j:j + pixel_size, i:i + pixel_size] = mean_color_bgr
    
    #Отображение изображения
    cv.imshow('Init image', image)
    cv.imshow('Output image', pixelation_image)
    cv.waitKey(0)

    #Сохранение изображения
    cv.imwrite(output_image, pixelation_image)

    #Освобождение ресурсов для последующей работы    
    cv.destroyAllWindows()
    
     
#Пишем функцию, которая запускает необходимый сценарий, исходя из аргументов командной строки
def main():
    args = cli_argument_parser()
    
    if args.mode == 'image':
        highgui_image_samples(args.image_path, args.output_image)
    elif args.mode == 'image_to_grayscale':
        image_to_grayscale(args.image_path, args.output_image)
    elif args.mode == 'image_resize':
        image_resize(args.image_path, args.output_image, args.value)
    elif args.mode == 'sepia':
        sepia(args.image_path, args.output_image)
    elif args.mode == 'vignette':
        vignette(args.image_path, args.output_image)
    elif args.mode == 'pixelation':
        pixelation(args.image_path, args.output_image, int(args.pixel_size))
    else:
        raise 'Unsupported \'mode\' value'

if __name__ == '__main__':
    sys.exit(main() or 0)

