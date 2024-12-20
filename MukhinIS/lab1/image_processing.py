import cv2 as cv
import argparse
import sys
import random
import numpy as np
from pathlib import Path


def pixelization(image, x1, y1, x2, y2, num_blocks):
    height, width = abs(x2 - x1), abs(y2 - y1)
    tmp = image[min(y1, y2):min(y1, y2) + width, min(x1, x2):min(x1, x2) + height]
    tmp1 = np.array_split(tmp, num_blocks, axis=0)
    blocks = []
    for elem in tmp1:
        blocks.extend(np.array_split(elem, num_blocks, axis=1))
    for bl in blocks:
        avr_color = bl.mean(axis=(0, 1)).astype(int)
        bl[:, :] = avr_color

    image[min(y1, y2):min(y1, y2) + width, min(x1, x2):min(x1, x2) + height] = tmp

    return image
    

def resize(image, coef):
    original_height, original_width = image.shape[:2]

    new_width = int(original_width * coef)
    new_height = int(original_height * coef)

    x_indices = np.linspace(0, original_width - 1, new_width).astype(int)
    y_indices = np.linspace(0, original_height - 1, new_height).astype(int)

    resized_image = image[np.ix_(y_indices, x_indices)]

    return resized_image


def sepia(image):
    sepia = np.zeros_like(image)
    new_b = np.clip(image[:, :, 0] * 0.131 + image[:, :, 1] * 0.534 + image[:, :, 2] * 0.272, 0, 255)
    new_g = np.clip(image[:, :, 0] * 0.168 + image[:, :, 1] * 0.686 + image[:, :, 2] * 0.349, 0, 255)
    new_r = np.clip(image[:, :, 0] * 0.189 + image[:, :, 1] * 0.769 + image[:, :, 2] * 0.393, 0, 255)
    sepia[:] = np.stack((new_b, new_g, new_r), axis=-1)
    return sepia


def vingette(image, radius):
    height, width = image.shape[:2]

    x_resultant_kernel = cv.getGaussianKernel(width, width / radius)
    y_resultant_kernel = cv.getGaussianKernel(height, height / radius)

    kernel = y_resultant_kernel * x_resultant_kernel.T
    mask = kernel / kernel.max()

    image_vignette = np.copy(image)

    for i in range(3):
        image_vignette[:,:,i] = image_vignette[:,:,i] * mask

    return image_vignette


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_path',
                        help='Path to image.',
                        type=Path,
                        required=True,
                        dest='image_path')
    parser.add_argument('-rc', '--resize_coef',
                        help='Coef for resize.',
                        type=int,
                        required=False,
                        default=0.5,
                        dest='rc')
    parser.add_argument('-nb', '--num_blocks',
                        help='Number of blocks in pixelization.',
                        type=int,
                        default=100,
                        required=False,
                        dest='nb')
    parser.add_argument('-r', '--radius',
                        help='Radius for vingette.',
                        type=float,
                        default=5.0,
                        required=False,
                        dest='rad')

    args = parser.parse_args()

    return args
    

def gray(image):
    return (image[:, :, 0] * 0.114 +
            image[:, :, 1] * 0.587 +
            image[:, :, 2] * 0.299).astype(np.uint8)


def read_image(image_path):
    if not image_path.is_file():
        raise ValueError("File doesn't exist." )

    return cv.imread(str(image_path))


def show_image(image_path, rc, nb, rad):
    img = read_image(image_path)
    img_pix = None
    while(True):
        cv.imshow('Original image', img)

        if img_pix is not None:
            cv.imshow('Pixelization', img_pix)

        k = cv.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv.imshow('Sepia', sepia(img))
        elif k == ord('v'):
            cv.imshow('Vingette', vingette(img, rad))
        elif k == ord('g'):
            cv.imshow('Gray scalled image', gray(img))
        elif k == ord('r'):
            cv.imshow('Resized', resize(img, rc))
        elif k == ord('p'):
            img_pix = np.copy(img)
            cv.namedWindow('Pixelization')
            r = cv.selectROI('Pixelization', img_pix, showCrosshair=False)
            pixelization(img_pix, r[0], r[1], r[0] + r[2], r[1] + r[3], nb)
    cv.destroyAllWindows()

def main():
    args = cli_argument_parser()
    show_image(args.image_path, args.rc, args.nb, args.rad)


if __name__=='__main__':
    sys.exit(main() or 0)