import argparse
import sys
import cv2 as cv
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'grayImage\', \'resolImage\', \'sepiaImage\', \'vignetteImage\', \'pixelImage\')',
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
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path')

    args = parser.parse_args()
    return args

def grayImage(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)
    gray_image = np.zeros_like(image, dtype=np.float32)

    gray_image[:, :, 0] = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray_image[:, :, 1] = gray_image[:, :, 0]
    gray_image[:, :, 2] = gray_image[:, :, 0]

    gray_image = gray_image.astype(np.uint8)

    cv.imshow('Base image', image)
    cv.imshow('Gray image', gray_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resolutionImage(image_path, new_width, new_height):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)

    x = np.arange(new_width) / (new_width - 1) * (image.shape[1] - 1)
    y = np.arange(new_height) / (new_height - 1) * (image.shape[0] - 1)
    x, y = np.meshgrid(x, y)

    x_coords = np.round(x).astype(int)
    y_coords = np.round(y).astype(int)

    resolution_image = image[y_coords, x_coords]

    cv.imshow('Base image', image)
    cv.imshow('Gray image', resolution_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def sepiaImage(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)
    sepia_image = np.zeros_like(image, dtype=np.float32)

    # Вычисляем значения для каждого канала (с учетом BGR)
    sepia_image[:, :, 0] = 0.393 * image[:, :, 2] + 0.769 * image[:, :, 1] + 0.189 * image[:, :, 0]
    sepia_image[:, :, 1] = 0.349 * image[:, :, 2] + 0.686 * image[:, :, 1] + 0.168 * image[:, :, 0]
    sepia_image[:, :, 2] = 0.272 * image[:, :, 2] + 0.534 * image[:, :, 1] + 0.131 * image[:, :, 0]

    sepia_image = np.clip(sepia_image, 0, 255)
    sepia_image = sepia_image.astype(np.uint8)
    sepia_image = cv.cvtColor(sepia_image, cv.COLOR_BGR2RGB)

    cv.imshow('Base image', image)
    cv.imshow('Sepia image', sepia_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def vignetteImage(image_path, radius, intensity):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)

    height = image.shape[0]
    width = image.shape[1]
    center_x = width // 2
    center_y = height // 2

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    norm = dist_from_center / (radius * max(center_x, center_y))
    norm = np.clip(norm, 0, 1)

    vignette = image * (1 - intensity * norm[..., np.newaxis])
    vignette = vignette.astype(np.uint8)

    cv.imshow('Base image', image)
    cv.imshow('Vignette', vignette)
    cv.waitKey(0)
    cv.destroyAllWindows()

def pixelImage(image_path, x, y, width, height, block_size):
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)
    pixel_image = image.copy()

    epsilon = pixel_image[y:y + height, x:x + width]

    blocks_x = width // block_size
    blocks_y = height // block_size
    for i in range(blocks_y):
        for j in range(blocks_x):
            block = epsilon[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

            avg_color = np.mean(block, axis=(0, 1))

            epsilon[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = avg_color.astype(np.uint8)

    pixel_image[y:y + height, x:x + width] = epsilon

    cv.imshow('Base image', image)
    cv.imshow('Region', pixel_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    args = cli_argument_parser()

    if args.mode == 'grayImage':
        grayImage(args.image_path)
    elif args.mode == 'resolImage':
        resolutionImage(args.image_path, 360, 240)
    elif args.mode == 'sepiaImage':
        sepiaImage(args.image_path)
    elif args.mode == 'vignetteImage':
        vignetteImage(args.image_path, 0.75, 1)
    elif args.mode == 'pixelImage':
        pixelImage(args.image_path, 100, 100, 200, 200, 20)
    else:
        raise 'Unsupported \'mode\' value'


if __name__ == '__main__':
    sys.exit(main() or 0)