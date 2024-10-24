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

def gray_img(image_path):
    image = read_img(image_path)
    
    height, width = image.shape[:2]
    gray_image = np.zeros((height, width), np.uint8)
    
    gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray_image = gray_image.astype(np.uint8)
    
    text = 'Gray image'
    show_img(text, image, gray_image)
    
def resolution_img(image_path, coef, op):
    src_image = read_img(image_path)
   
    height, width = src_image.shape[:2] 
    
    if op == 'm':
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
                

    text = 'Resolution image'
    show_img(text, src_image, resolution_image)

def sepia_img(image_path): 
    src_image = read_img(image_path)
    
    height, width = src_image.shape[:2]
    sepia_image = np.zeros((height, width, 3), np.uint8)

    gray = 0.299 * src_image[:, :, 2] + 0.587 * src_image[:, :, 1] + 0.114 * src_image[:, :, 0]
    sepia_image[:, :, 0] = np.clip(gray - 30, 0, 255)
    sepia_image[:, :, 1] = np.clip(gray + 15, 0, 255)
    sepia_image[:, :, 2] = np.clip(gray + 40, 0, 255)
              
    text = 'Sepia image'
    show_img(text, src_image, sepia_image)
    
def vignette_img(image_path, radius): 
    src_image = read_img(image_path)
    
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
    
    text = 'Vignette image'
    show_img(text, src_image, vig_img)
    
    
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
    
def pixel_img(image_path, block_size): 
    src_image = read_img(image_path)
    
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
    
    text = 'Pixel image'
    show_img(text, src_image, pixel_img)
    
    
def pixel2_img(image_path, block_size): 
    src_image = read_img(image_path)
    
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
    
    text = 'Pixel2 image'
    show_img(text, src_image, pixel_img2)
    


def main():
    args = cli_argument_parser()
    
    if args.mode == 'gray':
        gray_img(args.image_path)
    elif args.mode == 'res':
        resolution_img(args.image_path, args.coef, args.op)
    elif args.mode == 'sepia':
        sepia_img(args.image_path)
    elif args.mode == 'vig':
        vignette_img(args.image_path, args.radius)
    elif args.mode == 'pixel':
        pixel_img(args.image_path, args.block)
    elif args.mode == 'pixel2':
        pixel2_img(args.image_path, args.block)
    else:
        raise ValueError('Unsupported mode')


if __name__ == '__main__':
    sys.exit(main() or 0)


