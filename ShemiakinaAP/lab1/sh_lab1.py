#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import sys
import cv2 as cv
import numpy as np
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import random

x1, y1, x2, y2 = -1, -1, -1, -1

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    filters = ['grayshadows', 'resize', 'sepia', 
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
    parser.add_argument('-f', '--filter',
                        help='Filters: ' + ', '.join(filters),
                        type=str,
                        dest='filter')
    parser.add_argument('-p', '--parameter',
                        help='Parameter of the filter (default 2.0)\n' + 
                        '\n'.join(parameters_help),
                        type=float,
                        default=2.,
                        dest='parameter')
    parser.add_argument('-o', '--output',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    
    args = parser.parse_args()
    
    return args

def GrayShadows(image):
    B, G, R = cv.split(image)

    grayscale_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
    
    return grayscale_image

def Resize(image, coef):
    original_height, original_width = image.shape[:2]
    
    new_width = int(original_width * coef)
    new_height = int(original_height * coef)
    
    x_indices = np.linspace(0, original_width - 1, new_width).astype(int)
    y_indices = np.linspace(0, original_height - 1, new_height).astype(int)

    resized_image = image[np.ix_(y_indices, x_indices)]
    
    return resized_image

def Sepia(image):
    image = np.array(image)
    
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
    
    sepia_image = np.dot(image[..., :3], sepia_filter.T)   
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    
    return sepia_image

def Vignette(image,coef):
    height, width = image.shape[:2]

    vingnette_image = np.zeros_like(image)

    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            if distance < max_radius:
                vignette_strength = 1 - (distance / max_radius)** coef
            else:
                vignette_strength = 0
            vingnette_image[y, x] = image[y, x] * vignette_strength

    return vingnette_image

def Select_Area(image):
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
                cv.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.imshow("Area", image_copy)

    cv.setMouseCallback('Area', mouse_handle, param=image)
    cv.imshow("Area", image)
    while True:
        key = cv.waitKey(1)
        if key == ord('\r'): 
            break

    cv.destroyWindow('Area')

def Pixelization(image, block_size):
    global x1, y1, x2, y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    selected_area = image[y1:y2, x1:x2]
    area_height, area_width = selected_area.shape[:2]

    if (block_size > area_height or block_size > area_width):
        raise ValueError("Invalid block size")

    blocks_num_height = area_height // block_size
    blocks_num_width = area_width // block_size

    image_blocks = []
    for i in range(blocks_num_height):
        for j in range(blocks_num_width):
            block = selected_area[i * block_size:(i + 1) * block_size,
                                    j * block_size:(j + 1) * block_size]
            image_blocks.append(block)

    np.random.shuffle(image_blocks)

    new_image = []
    for i in range(blocks_num_height):
        new_row = np.hstack(image_blocks[i * blocks_num_width:(i + 1) * blocks_num_width])
        new_image.append(new_row)

    output_pixelized = np.vstack(new_image)
    pixelized_height, pixelized_width = output_pixelized.shape[:2]

    image[y1:y1 + pixelized_height, x1:x1 + pixelized_width] = output_pixelized

    return image

def main():
    args = argument_parser()
    
    image = cv.imread(args.image_path)
    if image is None:
        print("Warning: Failed to load image. Check the path.")
        return

    if args.filter == 'grayshadows':
        new_image = GrayShadows(image)
    elif args.filter == 'resize':
        new_image = Resize(image, args.parameter)
    elif args.filter == 'sepia':
        new_image = Sepia(image)
    elif args.filter == 'vignette':
        new_image = Vignette(image, args.parameter)
    elif args.filter == 'pixelization':
        Select_Area(image)
        new_image = Pixelization(image, int(args.parameter))
        
    cv.imshow('Original', cv.imread(args.image_path))
    cv.imwrite(args.output_image, new_image)
    cv.imshow('Changed image', new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)

# In[ ]:




