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
    args = parser.parse_args()

    return args

def gray_img(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
        
    src_image = cv.imread(image_path)
    
    height, width = src_image.shape[:2]
    gray_image = np.zeros((height, width), np.uint8)
    
    for y in range(height):
        for x in range(width):
            r, g, b = src_image[y, x]
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            gray_image[y, x] = gray_value 

    cv.imshow('Image', src_image)
    cv.imshow('Gray image', gray_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def resolution_img(image_path):
    if image_path is None:
       raise ValueError('Empty path to the image')
       
    src_image = cv.imread(image_path)
   
    height, width = src_image.shape[:2] 
    new_height = int(height/8)
    new_width = int(width/8)
    resolution_image = np.zeros((new_height, new_width, 3), np.uint8) 

    for y in range(new_height):
        for x in range(new_width):
          src_x = int(x * src_image.shape[1] / new_width)
          src_y = int(y * src_image.shape[0] / new_height)
          resolution_image[y, x] = src_image[src_y, src_x]
          
    cv.imshow('Image', src_image)
    cv.imshow('Resolution image', resolution_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def sepia_img(image_path): 
    if image_path is None:
       raise ValueError('Empty path to the image')
       
    src_image = cv.imread(image_path)
    
    height, width = src_image.shape[:2]
    sepia_image = np.zeros((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            r, g, b = src_image[y, x]
            gray = 0.299 * r + 0.587 * g + 0.114 * b 
            sepia_image[y, x, 0] = np.clip(int(gray) - 30, 0, 255)
            sepia_image[y, x, 1] = np.clip(int(gray) + 15, 0, 255)
            sepia_image[y, x, 2] = np.clip(int(gray) + 40, 0, 255)
              
    cv.imshow('Image', src_image)
    cv.imshow('Sepia image', sepia_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def vignette_img(image_path): 
    if image_path is None:
       raise ValueError('Empty path to the image')
       
    src_image = cv.imread(image_path)
    
    radius=0.75
    
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
    
    cv.imshow('Image', src_image)
    cv.imshow('Vignette image', vig_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def pixel_img(image_path): 
    if image_path is None:
       raise ValueError('Empty path to the image')
       
    src_image = cv.imread(image_path)
    
    x = 300
    y = 125
    width = 400
    height = 400
    block_size = 10
    
    pixel_img = np.zeros((src_image.shape[0], src_image.shape[1], 3), np.uint8)
    np.copyto(pixel_img, src_image)
    
    roi = pixel_img[y:y+height, x:x+width]
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
          block = roi[i:i+block_size, j:j+block_size]
          color = np.mean(block, axis=(0, 1)).astype(np.uint8)
          roi[i:i+block_size, j:j+block_size] = color

    pixel_img[y:y+height, x:x+width] = roi
    
    cv.imshow('Image', src_image)
    cv.imshow('Pixel', pixel_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def pixel2_img(image_path): 
    if image_path is None:
       raise ValueError('Empty path to the image')
       
    src_image = cv.imread(image_path)
    
    x = 300
    y = 125
    width = 400
    height = 400
    block_size = 100
    
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
    
    cv.imshow('Image', src_image)
    cv.imshow('Pixel2', pixel_img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
    


def main():
    args = cli_argument_parser()
    
    if args.mode == 'gray':
        gray_img(args.image_path)
    elif args.mode == 'res':
        resolution_img(args.image_path)
    elif args.mode == 'sepia':
        sepia_img(args.image_path)
    elif args.mode == 'vig':
        vignette_img(args.image_path)
    elif args.mode == 'pixel':
        pixel_img(args.image_path)
    elif args.mode == 'pixel2':
        pixel2_img(args.image_path)
    else:
        raise ValueError('Unsupported mode')


if __name__ == '__main__':
    sys.exit(main() or 0)

