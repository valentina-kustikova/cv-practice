import argparse
import sys
import cv2 as cv
import numpy as np
import random

x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False 

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\', \'convert_gray_color_filter\', \'resolution_change_filter\', \'sepia_filter\', \'pixelization_filter\', \'vignette_filter\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('--value', 
                        help='Resize factor for average_resize mode',
                        type=float, 
                        dest='value')
    parser.add_argument('--x1', 
                        help='Set x1 from the interval [x1,x2].',
                        type=int, 
                        dest='x1')
    parser.add_argument('--x2', 
                        help='Set x2 from the interval [x1,x2].',
                        type=int, 
                        dest='x2')
    parser.add_argument('--y1', 
                        help='Set y1 from the interval [y1,y2].',
                        type=int,
                        dest='y1' )
    parser.add_argument('--y2', 
                        help='Set y2 from the interval [y1,y2].',
                        type=int, 
                        dest='y2')
    parser.add_argument('--pix', 
                        help='Set pixel size.',
                        type=int,
                        dest='pixel_size')
    parser.add_argument('--radius', 
                        help='Set radius for vegnette filter.',
                        type=int,
                        dest='radius')
    
    args = parser.parse_args()
    return args

def highgui_image_samples(image_path, output_image):
    image = readImage(image_path)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def readImage(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    cv.imshow('Original image', image)
    return image

def writeImage(output_image, result_image):
    cv.imwrite(output_image, result_image)
    cv.imshow('Processed image', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def gray_filter(image):
    height, width, nchannels = image.shape

    result_image = np.zeros((height, width), np.uint8)
    result_image = image.copy()
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]
    coeff_gray = B*0.114 + G*0.587 + R*0.299
    for k in range(nchannels):
        result_image[:, :, k] = coeff_gray
    return result_image

def resize_filter(image, value):
    if value <= 0:
        raise ValueError('Value must be greater than zero')  
    height, width, nchannels = image.shape
    new_width = int(width * value)
    new_height = int(height * value)
    
    x_ind = np.floor(np.arange(new_width) / value).astype(int)
    y_ind = np.floor(np.arange(new_height) / value).astype(int)

    result_image = image[y_ind[:, None], x_ind]

    return result_image

def apply_sepia(image):
    height, width, nchannels = image.shape

    result_image = np.zeros((height, width, nchannels), np.uint8 )
    result_image[:, :, 2] = np.clip(0.393 * image[:, :, 2] + 0.769 * image[:, :, 1] + 0.189 * image[:, :, 0], 0, 255)
    result_image[:, :, 1] = np.clip(0.349 * image[:, :, 2] + 0.686 * image[:, :, 1] + 0.168 * image[:, :, 0], 0, 255)
    result_image[:, :, 0] = np.clip(0.272 * image[:, :, 2] + 0.534 * image[:, :, 1] + 0.131 * image[:, :, 0], 0, 255)
    return result_image

def apply_vignette(image, radius):
    height, width= image.shape[:2]

    centr_x = int(width / 2)
    centr_y = int(height / 2)
    y_indices, x_indices = np.indices((height, width))
    dist = np.sqrt((x_indices - centr_x) ** 2 + (y_indices - centr_y) ** 2)
    coef = 1 - np.minimum(1, dist / radius)
    result_image = (image * coef[:, :, np.newaxis]).astype(np.uint8)

    return result_image

def apply_pixel(image, pixel_size):
    nchannels = image.shape[2]
    display_image_with_rectangle(image)

    result_image = image.copy()
    for x in range(x1, x2, pixel_size):
        for y in range(y1, y2, pixel_size):
            end_x = min(x + pixel_size, x2)
            end_y = min(y + pixel_size, y2)

            region = image[y:end_y, x:end_x]
            mean_color_bgr = np.zeros(nchannels, dtype=int)

            for k in range(nchannels):
                mean_color_bgr[k] = int(np.mean(region[:, :, k]))

            result_image[y:end_y, x:end_x] = mean_color_bgr
    return result_image


        
def convert_gray_color_filter(image_path, output_image):
    image = readImage(image_path)
    result_image = gray_filter(image)
    writeImage(output_image, result_image)

def resolution_change_filter(image_path, output_image, value):
    image = readImage(image_path)
    result_image = resize_filter(image,value)
    writeImage(output_image, result_image) 

def sepia_filter(image_path, output_image):
    image = readImage(image_path)
    result_image = apply_sepia(image)
    writeImage(output_image, result_image)

def display_image_with_rectangle(image):
    global x1, y1, x2, y2

    cv.namedWindow('Image')
    cv.setMouseCallback('Image', draw_rectangle)

    while True:
        img_copy = image.copy()
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv.imshow('Image', img_copy)

        key = cv.waitKey(1)
        if key == ord('q'): 
            break

    cv.destroyWindow('Image')
    
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y  

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y  

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y  


def pixelization_filter(image_path, output_image, pixel_size):
    global x1, y1, x2, y2
    image = readImage(image_path)
    result_image = apply_pixel(image, pixel_size)
    writeImage(output_image, result_image)
 
def vignette_filter(image_path, output_image, radius):
    image = readImage(image_path)
    result_image = apply_vignette(image, radius)
    writeImage(output_image, result_image)

def main():
    args = cli_argument_parser()
    
    if args.mode == 'image':
        highgui_image_samples(args.image_path, args.output_image)
    elif args.mode == 'convert_gray_color_filter':
        convert_gray_color_filter(args.image_path, args.output_image)
    elif args.mode == 'resolution_change_filter':
        if args.value is None:
            raise ValueError('The value parameter must be provided for average_resize mode')
        resolution_change_filter(args.image_path, args.output_image, args.value)
    elif args.mode == 'sepia_filter':
        sepia_filter(args.image_path, args.output_image)
    elif args.mode == 'pixelization_filter':
        if args.pixel_size is None:
            raise ValueError('The pixel_size parameter must be provided for pixelate mode')
        pixelization_filter(args.image_path, args.output_image, args.pixel_size)
    elif args.mode == 'vignette_filter':
        vignette_filter(args.image_path, args.output_image, args.radius)
    else:
        raise ValueError('Unsupported \'mode\' value')


if __name__ == '__main__':
    sys.exit(main() or 0)