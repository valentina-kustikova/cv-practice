import cv2 as cv
import argparse
import sys
import numpy as np
import random


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
                        help='Mode (image, video, imgproc)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-c', '--coef',
                        help='Input coef',
                        type=int,
                        dest='coef')
    parser.add_argument('-op', '--operation',
                        help='Input operation',
                        type=str,
                        dest='op')
    parser.add_argument('-r', '--radius',
                        help='Input radius',
                        type=float,
                        dest='radius')
    parser.add_argument('-b', '--block',
                        help='Input block size',
                        type=int,
                        dest='block')
    args = parser.parse_args()

    return args

def read_img(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    src_image = cv.imread(image_path)
    return src_image

def show_img(text, image, image2):
    if image is None:
        raise ValueError('Empty path to the image')
    cv.imshow('Image', image)
    cv.imshow(text, image2)
    cv.waitKey(0)
    cv.destroyAllWindows()

def gray_img(src_image):
    
    height, width = src_image.shape[:2]
    gray_image = np.zeros((height, width), np.uint8)
    
    gray_image = 0.299 * src_image[:, :, 0] + 0.587 * src_image[:, :, 1] + 0.114 * src_image[:, :, 2]
    gray_image = gray_image.astype(np.uint8)
    return gray_image
    
def resolution_img(src_image, coef, op):
   
    height, width = src_image.shape[:2] 
    
    if op == 's':
        new_height = int(height/coef)
        new_width = int(width/coef)
    if op == 'b':
        new_height = int(height*coef)
        new_width = int(width*coef)
        
    resolution_image = np.zeros((new_height, new_width, 3), np.uint8) 

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * src_image.shape[1] / new_width)
            src_y = int(y * src_image.shape[0] / new_height)
            resolution_image[y, x] = src_image[src_y, src_x]
                

    return resolution_image

def sepia_img(src_image): 
    
    height, width = src_image.shape[:2]
    sepia_image = np.zeros((height, width, 3), np.uint8)

    gray = 0.299 * src_image[:, :, 2] + 0.587 * src_image[:, :, 1] + 0.114 * src_image[:, :, 0]
    sepia_image[:, :, 0] = np.clip(gray - 30, 0, 255)
    sepia_image[:, :, 1] = np.clip(gray + 15, 0, 255)
    sepia_image[:, :, 2] = np.clip(gray + 40, 0, 255)
              
    return sepia_image
    
def vignette_img(src_image, radius): 
    
    #radius=0.75

    vig_img = np.zeros((src_image.shape[0], src_image.shape[1], 3), np.uint8)
    np.copyto(vig_img, src_image)
    
    height, width = vig_img.shape[:2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = int(width/2), int(height/2)
    distance_from_center = np.sqrt((x - center_x)*(x - center_x) + (y - center_y)*(y - center_y))
    normalized_distance = distance_from_center / np.max(distance_from_center)
    mask = np.clip(1 - (normalized_distance / radius)*(normalized_distance / radius), 0, 1)

    for i in range(3):
        vig_img[:, :, i] = vig_img[:, :, i] * mask
    
    return vig_img
    
    
def area(image):
    def mouse_click(event, x, y, flags, param):
        nonlocal new_x, new_y, new_width, new_height
        if event == cv.EVENT_LBUTTONDOWN:
            new_x = x
            new_y = y
        elif event == cv.EVENT_LBUTTONUP:
            new_width = x - new_x
            new_height = y - new_y
    
    new_x = 0
    new_y = 0
    new_width = 0
    new_height = 0
    
    cv.imshow('Area', image)
    cv.setMouseCallback('Area', mouse_click)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return (new_x, new_y, new_width, new_height)
    
def pixel_img(src_image, block_size): 
    x, y, width, height = area(src_image)
    
    pixel_img = np.zeros((src_image.shape[0], src_image.shape[1], 3), np.uint8)
    np.copyto(pixel_img, src_image)
    
    roi = pixel_img[y:y+height, x:x+width]
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
          block = roi[i:i+block_size, j:j+block_size]
          color = np.mean(block, axis=(0, 1)).astype(np.uint8)
          roi[i:i+block_size, j:j+block_size] = color

    pixel_img[y:y+height, x:x+width] = roi
    
    return pixel_img
    
    
def pixel2_img(src_image, block_size): 
    
    x, y, width, height = area(src_image)
    
    pixel_img2 = np.zeros((src_image.shape[0], src_image.shape[1], 3), np.uint8)
    np.copyto(pixel_img2, src_image)
    
    roi = pixel_img2[y:y+height, x:x+width]
          
    block_coords = []
    for y_block in range(0, roi.shape[0] - block_size + 1, block_size):
        for x_block in range(0, roi.shape[1] - block_size + 1, block_size):
            block_coords.append((x_block, y_block))
                  
    random.shuffle(block_coords)
   
    for i in range(0, len(block_coords), 2): 
        if i + 1 < len(block_coords):
            block1_coords = block_coords[i]
            block2_coords = block_coords[i + 1]
            x1, y1 = block1_coords
            x2, y2 = block2_coords
            
            temp_block1 = roi[y1:y1+block_size, x1:x1+block_size].copy() 
            temp_block2 = roi[y2:y2+block_size, x2:x2+block_size].copy()

            roi[y1:y1+block_size, x1:x1+block_size] = temp_block2
            roi[y2:y2+block_size, x2:x2+block_size] = temp_block1

    pixel_img2[y:y+height, x:x+width] = roi
    
    return pixel2_img
    


def main():
    args = cli_argument_parser()
    
    src_image = read_img(args.image_path)
    
    
    if args.mode == 'gray':
        new_image = gray_img(src_image)
        text = 'Gray image'
    elif args.mode == 'res':
        new_image = resolution_img(src_image, args.coef, args.op)
        text = 'Resolution image'
    elif args.mode == 'sepia':
        new_image = sepia_img(src_image)
        text = 'Sepia image'
    elif args.mode == 'vig':
        new_image = vignette_img(src_image, args.radius)
        text = 'Vignette image'
    elif args.mode == 'pixel':
        new_image = pixel_img(src_image, args.block)
        text = 'Pixel image'
    elif args.mode == 'pixel2':
        new_image = pixel2_img(src_image, args.block)
        text = 'Pixel2 image'
    else:
        raise ValueError('Unsupported mode')
    
    show_img(text, src_image, new_image)


if __name__ == '__main__':
    sys.exit(main() or 0)
