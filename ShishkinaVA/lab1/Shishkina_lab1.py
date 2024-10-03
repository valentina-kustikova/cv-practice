import argparse
import sys
import cv2 as cv
import numpy as np

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
    
    args = parser.parse_args()
    return args

def highgui_image_samples(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    height, width, nchannels = image.shape
    cv.imshow('Init image', image)
    cv.waitKey(0)
    cv.imwrite(output_image, image)
    cv.destroyAllWindows()

def convert_gray_color_filter(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')

    height, width, nchannels = image.shape

    result_image = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            coeff_gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            result_image[i, j] = coeff_gray

    cv.imwrite(output_image, result_image)

    cv.imshow('Original image', image)
    cv.imshow('Processed image (convert_gray_color_filter)', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def resolution_change_filter(image_path, output_image, value):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    if value <= 0:
        raise ValueError('Value must be greater than zero')

    height, width, nchannels = image.shape
    new_width = int(width*value)
    new_height = int(height*value)
    result_image = np.zeros((new_height, new_width, nchannels),  np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            tmp_i = int(i/value)
            tmp_j = int(j/value)

            result_image[i, j] = image[tmp_i, tmp_j]

    cv.imwrite(output_image, result_image)

    cv.imshow('Original image', image)
    cv.imshow('Processed image (resolution_change_filter)', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def sepia_filter(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    height, width, nchannels = image.shape

    result_image = np.zeros((height, width, nchannels), np.uint8 )

    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            tmp_r = int(0.393*r + 0.769*g + 0.189*b)
            tmp_g = int(0.349*r + 0.686*g + 0.168*b)
            tmp_b = int(0.272*r + 0.534*g + 0.131*b)
            result_image[i, j] = [min(tmp_b, 255), min(tmp_g, 255), min(tmp_r, 255)]

    cv.imwrite(output_image, result_image)

    cv.imshow('Original image', image)
    cv.imshow('Processed image (sepia_filter)', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def pixelization_filter(image_path, output_image, x1, x2, y1, y2, pixel_size):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    height, width, nchannels = image.shape

    result_image = np.zeros((height, width, nchannels), np.uint8 )
    result_image = image.copy()

    for x in range(x1, x2, pixel_size):
        for y in range(y1, y2, pixel_size):
            end_y = y + pixel_size
            end_x = x + pixel_size
            region = image[x:end_x, y:end_y]

            mean_color_bgr = np.zeros(nchannels, dtype=int)
            for k in range(nchannels):
                mean_color_bgr[k] = np.mean(region[:, :, k])

            result_image[ x:end_x, y:end_y] = mean_color_bgr

    cv.imwrite(output_image, result_image)

    cv.imshow('Original image', image)
    cv.imshow('Processed image (pixelization_filter)', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()


def vignette_filter(image_path, output_image):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    height, width, nchannels = image.shape

    result_image = np.zeros((height, width, nchannels), np.uint8 )
    centr_y = int(height/2)
    centr_x = int(width/2)
    for i in range(height):
        for j in range(width):
            delta_j = 1 - (abs(j - centr_x) / centr_x)
            delta_i = 1 - (abs(i - centr_y) / centr_y)
            result_image[i,j] = image[i,j]*delta_i*delta_j

    cv.imwrite(output_image, result_image)

    cv.imshow('Original image', image)
    cv.imshow('Processed image (vignette_filter)', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

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
        pixelization_filter(args.image_path, args.output_image, args.x1, args.x2, args.y1, args.y2, args.pixel_size)
    elif args.mode == 'vignette_filter':
        vignette_filter(args.image_path, args.output_image)
    else:
        raise ValueError('Unsupported \'mode\' value')


if __name__ == '__main__':
    sys.exit(main() or 0)