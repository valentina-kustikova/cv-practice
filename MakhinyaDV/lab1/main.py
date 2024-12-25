#Импортируем необходимые библиотеки
import cv2 as cv
import numpy as np
import argparse
import sys

x1, y1, x2, y2 = -1, -1, -1, -1
def cli_argument_parser():
    parser = argparse.ArgumentParser()

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


def image_to_grayscale(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def image_resize(image, value):
    new_height, new_width = int(image.shape[0] * value), int(image.shape[1] * value)

    new_y = (np.arange(new_height) / value).astype(int)
    new_x = (np.arange(new_width) / value).astype(int)

    return image[new_y[:, None], new_x]


def sepia(image):
    sepia_filter = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_image = np.dot(image[..., :3], sepia_filter.T)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)


def vignette(image, radius):
    height, width = image.shape[:2]
    x, y = width // 2, height // 2
    y_coords, x_coords = np.indices((height, width))
    distance = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
    coef_vin = np.clip(1 - distance / radius, 0, 1)
    return (image * coef_vin[:, :, np.newaxis]).astype(np.uint8)

#Функция для пексилизации области изображения
def pixel_filter(image, x1, y1, x2, y2, pixel_size):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    pixel_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
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

    pixel_image[y1:y2, x1:x2] = reg
    return image

def callback(event, x, y, flags, param):
    global rect, drawing, x1, y1, x2, y2
    #drawing = False
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y  

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y  

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y 

        
def eventHandler(image, size):
    global img, rect, drawing, pixelate_size

    cv.namedWindow('Image')
    cv.setMouseCallback('Image',  callback)

    while True:
        img_copy = image.copy()
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv.rectangle(img_copy, (x1, y1), (x2, y2), (128, 255, 0), 2) 
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
        copy_im = image.copy();
        eventHandler(copy_im, args.pixelate_size)
        filtr_image = pixel_filter(copy_im, x1, y1, x2, y2, args.pixelate_size)
    else:
        raise 'Unsupported \'mode\' value'

    cv.imshow('Init image', image)
    cv.imshow('Output image', filtr_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)