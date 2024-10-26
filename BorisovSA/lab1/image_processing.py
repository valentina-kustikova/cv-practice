"""Image processing using OpenCV library."""

import argparse
import sys
import cv2 as cv
import numpy as np
from argparse import RawTextHelpFormatter


def grayscale(image):
    B, G, R = cv.split(image)
    grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    
    return grayscale_image

def resize(image, coef):
    original_height, original_width = image.shape[:2]
    
    new_width = int(original_width * coef)
    new_height = int(original_height * coef)
    
    x_indices = np.linspace(0, original_width - 1, new_width).astype(int)
    y_indices = np.linspace(0, original_height - 1, new_height).astype(int)

    resized_image = image[np.ix_(y_indices, x_indices)]
    
    return resized_image

def sepia(image):
    sepia_image = np.zeros_like(image)
    
    coef = 16.
    intensity = grayscale(image)
    
    new_R = np.clip(np.round(intensity + 2. * coef), 0, 255)
    new_G = np.clip(np.round(intensity + 0.5 * coef), 0, 255)
    new_B = np.clip(np.round(intensity - 1. * coef), 0, 255)
    
    sepia_image[:] = np.stack((new_B , new_G , new_R), axis=-1)
    
    return sepia_image

def vignette(image, coef):
    rows, cols = image.shape[:2]
    
    y_indices, x_indices = np.indices((rows, cols))
    
    center_x, center_y = cols // 2, rows // 2
    distance = np.sqrt((x_indices - center_x) ** 2 +
                       (y_indices - center_y) ** 2)

    vignette_value = np.exp(-coef * (distance ** 2) / 
                            (center_x**2 + center_y**2))
    
    vignette_image = image.astype(np.float32) * vignette_value[:, :,
                                                               np.newaxis]
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)
    
    return vignette_image

def pixelization(image, min_blocks_num):
    height, width = image.shape[:2]
    
    square_size = min(height, width) // 2

    start_row = (height - square_size) // 2
    start_col = (width - square_size) // 2

    center_square = image[start_row:start_row + square_size,
                      start_col:start_col + square_size]
    
    candidates = np.arange(min_blocks_num, square_size // 2)
    gcd_values = np.gcd(square_size, candidates)
    suitable_divisors = gcd_values[gcd_values >= min_blocks_num]
    
    blocks_num = np.min(suitable_divisors)
    split_square = np.split(center_square, blocks_num, axis=0)
    
    blocks = []
    for row in range(blocks_num):
        blocks.extend(np.split(split_square[row], blocks_num, axis=1))
        
    np.random.shuffle(blocks)
    new_image = []
    for row in range(blocks_num):
        new_image.append(np.hstack(blocks[row * blocks_num:
                                          row * blocks_num + blocks_num]))
        
    image[start_row:start_row + square_size,
                      start_col:start_col + square_size] = np.vstack(new_image)
    
    return image

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    filters = ['grayscale', 'resize', 'sepia', 
                           'vignette', 'pixelization']
    
    parameters_help = ['resize: Factor to resize the image',
                       'vignette: Strength of vignette effect',
                       'pixelization: Lower limit of the number '
                        'of pixels per square area length']
                        
    for i in range(len(parameters_help)):
        parameters_help[i] = ('- ' + parameters_help[i])
    
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('-f', '--filter',
                        help='Filters: ' + ', '.join(filters),
                        type=str,
                        dest='filter')
    parser.add_argument('-p', '--parameter',
                        help='Parameter of the filter (default 2.0 for all)\n' + 
                        '\n'.join(parameters_help),
                        type=float,
                        default=2.,
                        dest='parameter')
    
    args = parser.parse_args()
    
    if args.filter not in ['grayscale', 'resize', 'sepia', 
                           'vignette', 'pixelization']:
        parser.error('Invalid filter selected: ', args.filter)
    
    return args

def main():
    args = argument_parser()
    
    image = cv.imread(args.image_path)
    
    if args.filter == 'grayscale':
        new_image = grayscale(image)
    elif args.filter == 'resize':
        new_image = resize(image, args.parameter)
    elif args.filter == 'sepia':
        new_image = sepia(image)
    elif args.filter == 'vignette':
        new_image = vignette(image, args.parameter)
    elif args.filter == 'pixelization':
        new_image = pixelization(image, int(args.parameter))
        
    cv.imshow('Original image', cv.imread(args.image_path))
    cv.imwrite(args.output_image, new_image)
    cv.imshow('New image', new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)