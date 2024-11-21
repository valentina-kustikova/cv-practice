#Импортируем необходимые библиотеки
import cv2 as cv
import numpy as np
import argparse
import sys

x1, y1, x2, y2 = -1, -1, -1, -1
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
                        help='Mode (\'image\', \'gray_image\', \'resize_image\', \'sepia_image\', \'vignette_image\', \'pixel_image\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-v', '--value',
                        help='input value for filter',
                        type=float,
                        dest='value',
                        default='1')
    parser.add_argument('-r', '--radius',
                        help='radius',                   
                        type=int,
                        dest='radius',
                        default='150')
    parser.add_argument('-p', '--pixelate_size',
                        help='pixelate size',
                        type=int,
                        dest='pixelate_size',
                        default='15')    
    
    args = parser.parse_args()
    return args


#Функция перевода изображения в оттенки серого
def image_to_grayscale(image):
    height, width, nchannels = image.shape
    
    grey_image = np.zeros((height, width, 3), dtype=np.uint8)
            
    R = image[:,:,0] 
    G = image[:,:,1] 
    B = image[:,:,2] 
            
    grey_image = 0.2989 * R + 0.5870 * G + 0.1140 * B
    
    grey_image = image[:,:,0] 
    grey_image = image[:,:,1] 
    grey_image = image[:,:,2] 
    
    return grey_image
    

#Функция изменения разрешения изображения
def image_resize(image, value):
    height, width, nchannels = image.shape
    
    new_height = int(height*value)
    new_width = int(width*value)
    
    resize_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
             
    new_y = np.floor(np.arange(new_height) / value).astype(int)        
    new_x = np.floor(np.arange(new_width) / value).astype(int)

    resize_image = image[new_y[:, None], new_x]

    return resize_image


#Функция применения фотоэффекта сепии к изображению.
def sepia(image):
    height, width, nchannels = image.shape

    sepia_image = np.zeros((height, width, 3), dtype=np.uint8)

    B = image[:,:,0] 
    G = image[:,:,1] 
    R = image[:,:,2] 
    
    sepia_image[:,:,0] = np.clip(0.272 * R + 0.534 * G + 0.131 * B, 0, 255) 
    sepia_image[:,:,1] = np.clip(0.349 * R + 0.686 * G + 0.168 * B, 0, 255)
    sepia_image[:,:,2] = np.clip(0.393 * R + 0.769 * G + 0.189 * B, 0, 255)
    
    return sepia_image
 
    
#Функция применения фотоэффекта виньетки к изображению.
def vignette(image, radius):
    height, width, nchannels = image.shape

    vignette_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    height = image.shape[0]
    width = image.shape[1]

    x = width // 2
    y = height // 2
    y_value, x_value = np.indices((height, width))
    
    distance = np.sqrt((x_value - x) ** 2 + (y_value - y) ** 2)
    coef_vin = np.clip(1 - distance / radius, 0, 1)
    vignette_image = (image * coef_vin[:, :, np.newaxis]).astype(np.uint8)
           
    return vignette_image
    

#Функция пикселизации заданной прямоугольной области изображения

#Функция для пексилизации области изображения
def pixel_filter(image, x1, y1, x2, y2, pixel_size):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    reg = image[y1:y2, x1:x2]
    reg_h, reg_w = reg.shape[:2]

    new_h = reg_h // pixel_size
    new_w = reg_w // pixel_size

    for i in range(0, new_h):
        for j in range(0, new_w):
            
            start_y = i * pixel_size
            end_y = start_y + pixel_size
            start_x = j * pixel_size
            end_x = start_x + pixel_size
            end_y = min(end_y, reg_h)
            end_x = min(end_x, reg_w)
            
            pixel_block = reg[start_y:end_y, start_x:end_x]
            avg_color = pixel_block.mean(axis=(0, 1)).astype(int)
            reg[start_y:end_y, start_x:end_x] = avg_color

    image[y1:y2, x1:x2] = reg

#Функция, отвечающая за обработку события мыши для рисования прямоугольника и пикселизации выбранной области
def draw(event, x, y, flags, param):
    global rect, drawing, img, pixel_image, pixelate_size, x1, y1, x2, y2

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y  

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y  

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y 
        
        
#Сама функция пикселизации        
def pixel_image(image, size):
    global img, pixel_image, rect, drawing, pixelate_size

    cv.namedWindow('Image')
    cv.setMouseCallback('Image', draw)

    while True:
        img_copy = image.copy()
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv.imshow('Image', img_copy)

        key = cv.waitKey(1)
        if key == ord('q'): 
            break

    cv.destroyWindow('Image')


    
#Пишем функцию, которая запускает необходимый сценарий, исходя из аргументов командной строки, а также вызов чтения и вывода изображений
def main():
    args = cli_argument_parser()
    
    image_path = args.image_path
    if image_path is None:
       raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
   
    
    #if args.mode == 'image':
        #highgui_image_samples(args.image_path, args.output_image)
    if args.mode == 'gray_image':
        filtr_image = image_to_grayscale(image)
    elif args.mode == 'resize_image':
        filtr_image = image_resize(image, args.value)
    elif args.mode == 'sepia_image':
        filtr_image = sepia(image)
    elif args.mode == 'vignette_image':
        filtr_image = vignette(image, args.radius)
    elif args.mode == 'pixel_image':
        pixel_image(image, args.pixelate_size)
        filtr_image = pixel_filter(image, x1, y1, x2, y2, args.pixelate_size)
    else:
        raise 'Unsupported \'mode\' value'
        
    cv.imshow('Init image', image)
    cv.imshow('Output image', filtr_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
    
