"""Image processing using OpenCV library."""

import argparse
import sys
import cv2 as cv
import numpy as np
from argparse import RawTextHelpFormatter

x1, y1, x2, y2 = -1, -1, -1, -1

def grayscale(image):
    B, G, R = cv.split(image)
    grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    
    return grayscale_image

def resize(image, coef):
    if (coef <= 0):
        raise ValueError("Invalid coefficient")
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

def vignette(image, radius):
    if (radius < 0):
        raise ValueError("Invalid radius")
    coef = 10

    rows, cols = image.shape[:2]
    
    y_indices, x_indices = np.indices((rows, cols))
    
    center_x, center_y = cols // 2, rows // 2
    distance = np.sqrt((x_indices - center_x) ** 2 +
                       (y_indices - center_y) ** 2)
    
    is_inside = (distance <= radius).astype(bool)
    vignette_value = np.exp(-coef * ((distance - radius) ** 2) / 
                            (center_x**2 + center_y**2)) * (~is_inside) + is_inside
    
    vignette_image = image.astype(np.float32) * vignette_value[:, :,
                                                               np.newaxis]
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)
    
    return vignette_image

def select_an_area(image):
    global x1, y1, x2, y2
    cv.namedWindow('Area')

    def mouse_handle(event, x, y, flags, param):
        global x1, y1, x2, y2

        if event == cv.EVENT_LBUTTONDOWN:
            x1, y1 = x, y
        elif event == cv.EVENT_MOUSEMOVE:
            if flags == cv.EVENT_FLAG_LBUTTON:
                x2, y2 = x, y
                image_copy = param.copy()
                cv.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.imshow("Area", image_copy)

    cv.setMouseCallback('Area', mouse_handle, param=image)
    cv.imshow("Area", image)
    while True:
        key = cv.waitKey(1)
        if key == ord('\r'): 
            break

    cv.destroyWindow('Area')

def pixelization(image, block_size):
    global x1, x2, y1, y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    selected_region = image[y1:y2, x1:x2]
    height, width = selected_region.shape[:2]

    if (block_size > height or block_size > width):
        raise ValueError("Invalid block size")

    blocks_num_height = height // block_size
    blocks_num_width = width // block_size

    blocks = []
    for i in range(blocks_num_height):
        for j in range(blocks_num_width):
            block = selected_region[i * block_size:(i + 1) * block_size,
                                    j * block_size:(j + 1) * block_size]
            blocks.append(block)

    np.random.shuffle(blocks)

    new_image = []
    for i in range(blocks_num_height):
        new_row = np.hstack(blocks[i * blocks_num_width:(i + 1) * blocks_num_width])
        new_image.append(new_row)
        
    pixelized_region = np.vstack(new_image)
    height, width = pixelized_region.shape[:2]
    
    image[y1:y1 + height, x1:x1 + width] = pixelized_region
    
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

def read_image(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)

    if image is None:
        raise ValueError('Unable to load image')
    cv.imshow('Original image', cv.imread(image_path))

    return image

def main():
    args = argument_parser()
    image = read_image(args.image_path)
    
    if args.filter == 'grayscale':
        new_image = grayscale(image)
    elif args.filter == 'resize':
        if args.parameter is None:
            raise ValueError("Required parameter not specified")
        new_image = resize(image, args.parameter)
    elif args.filter == 'sepia':
        new_image = sepia(image)
    elif args.filter == 'vignette':
        if args.parameter is None:
            raise ValueError("Required parameter not specified")
        new_image = vignette(image, args.parameter)
    elif args.filter == 'pixelization':
        if args.parameter is None:
            raise ValueError("Required parameter not specified")
        select_an_area(image)
        new_image = pixelization(image, int(args.parameter))
    
    cv.imwrite(args.output_image, new_image)
    cv.imshow('New image', new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
