import cv2 as cv
import argparse
import sys
import numpy as np
import random

def argument_parser():
   #"""Парсер командной строки для получения параметров"""
    parser = argparse.ArgumentParser(prog='lab1 - image processing',
                                     description="This laboratory work is devoted to basic operations on images.")

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
                        help='Mode (gray, sepia, resize, vig, pixel)',
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

def load_image(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    return cv.imread(image_path)

def show_image_private(image, text):
    window_name = text

    if image is not None:
        cv.imshow(window_name, image)
    else:
        raise ValueError('Empty image')
    

def show_images(original_image, result_image):
    show_image_private(original_image, 'original image')
    show_image_private(result_image,  'result_image')
    cv.waitKey(0)
    cv.destroyAllWindows()

def gray_filter(original_image):

    if original_image is None:
        raise ValueError('Empty image')
    
    result_image = 0.299 * original_image[:, :, 0] + 0.587 * original_image[:, :, 1] + 0.114 * original_image[:, :, 2]
    return result_image.astype(np.uint8)

def sepia_filter(original_image):

    height, width = original_image.shape[:2]
    result_image = np.zeros((height, width, 3), np.uint8)

    B = original_image[:, :, 0]
    G = original_image[:, :, 1]
    R = original_image[:, :, 2]

    result_image[:, :, 0] = np.clip(0.272 * R + 0.534 * G + 0.131 * B, 0, 255)
    result_image[:, :, 1] = np.clip(0.349 * R + 0.686 * G + 0.168 * B, 0, 255)
    result_image[:, :, 2] = np.clip(0.393 * R + 0.769 * G + 0.189 * B, 0, 255)

    return result_image

def resize(original_image, scale):
    
    height, width, number_channels = original_image.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    result_image = np.zeros((new_height, new_width, number_channels) ,dtype=np.uint8)

    for i in range (new_height):
        for j in range(new_width):
            result_image[i, j] = original_image[int(i/scale), int(j/scale)]
    
    return result_image

def vignette_img(original_image, radius):
# Extracting the height and width of an image 
    rows, cols = original_image.shape[:2]
    
    # generating vignette mask using Gaussian 
    # resultant_kernels
    X_resultant_kernel = cv.getGaussianKernel(cols, radius)
    Y_resultant_kernel = cv.getGaussianKernel(rows, radius)
    
    #generating resultant_kernel matrix 
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    
    #creating mask and normalising by using np.linalg
    # function

    mask = resultant_kernel / resultant_kernel.max()
    
    result_image = np.copy(original_image)
    
    # applying the mask to each channel in the input image
    for i in range(3):
        result_image[:,:,i] = result_image[:,:,i] * mask

    return result_image


def get_square(any_image):
    if any_image is None :
        raise ValueError('Empty image')

    new_x, new_y, width, height = 0, 0, 0, 0

    def mouse_click(event, x, y, flags, param):
        nonlocal new_x, new_y, width, height
        if event == cv.EVENT_LBUTTONDOWN:
            new_x, new_y = x, y
        elif event == cv.EVENT_LBUTTONUP:
            width = x - new_x
            height = y - new_y

    cv.imshow('Area', any_image)
    cv.setMouseCallback('Area', mouse_click)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (new_x, new_y, width, height)

def pix_filter(original_image, new_x, new_y, width, height, block_size):
    if original_image is None :
        raise ValueError('Empty image')
    if width == 0 or height == 0 :
        raise ValueError('Empty area')
    if block_size == 0:
        raise ValueError('Empty block')
    if block_size > height or block_size > width:
        raise ValueError("Invalid block size")
    
    result_image = np.copy(original_image)

    # we have image, and coords of area, width and heigth of this area, size of blocks ( at this moment - line)

    for i in range(new_y, new_y + height, block_size):
        for j in range(new_x, new_x + width, block_size):
            # Извлечение блока
            block = result_image[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            result_image[i:i + block_size, j:j + block_size] = avg_color
    return result_image


def main():
    args = argument_parser()
    original_image = load_image(args.image_path)
    if args.mode == 'gray':
        result_image = gray_filter(original_image)

    elif args.mode == 'sepia':
        result_image = sepia_filter(original_image)
        
    elif args.mode == 'resize':
        result_image = resize(original_image, args.coef)

    elif args.mode == 'vig':
        result_image = vignette_img(original_image, args.radius)

    elif args.mode == 'pixel':
        new_x, new_y, width, height = get_square(original_image)
        result_image = pix_filter(original_image, new_x, new_y, width, height, args.block)

    show_images(original_image, result_image)


if __name__ == '__main__':
    sys.exit(main() or 0)

