import cv2
import cv2 as cv
import numpy as np
import argparse
import sys

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task',
                        help='Available tasks: (idle, sepia, grayscale, resize, vignette, pixelate)',
                        type=str,
                        dest='task',
                        default='idle')
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')
    parser.add_argument('--width',
                        help='New width (in resize mode)',
                        type=int,
                        default=-1,
                        dest='width')
    parser.add_argument('--height',
                        help='New height (in resize mode)',
                        type=int,
                        default=-1,
                        dest='height')
    parser.add_argument('-r', '--radius',
                        help='Radius of vignette',
                        type=float,
                        default=-1,
                        dest='radius')
    parser.add_argument('-p', '--pixel_size',
                        help='Size of pixelation element',
                        type=int,
                        default=-1,
                        dest='pixel_size')
    args = parser.parse_args()

    return args



def gray_filter(image):
    return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)

def resize(image, w, h):
    y = np.floor(np.arange(h) / h * image.shape[0]).astype(int)
    x = np.floor(np.arange(w) / w * image.shape[1]).astype(int)
    return image[y[:, None], x]

def vignette(image, r):
    rows, cols = image.shape[:2]
    x_kernel = cv2.getGaussianKernel(cols, r)
    y_kernel = cv2.getGaussianKernel(rows, r)
    kernel = y_kernel * x_kernel.T
    mask = kernel / kernel.max()
    processed_img = image.copy()

    for i in range(3):
        processed_img[:, :, i] = processed_img[:, :, i] * mask

    return processed_img

def get_area(image, pixel_size):
    x, y, w, h = 0, 0, 0, 0

    def mouse_click(event, e_x, e_y, flags, param):
        nonlocal x, y, w, h, image, pixel_size
        if event == cv.EVENT_LBUTTONDOWN:
            x, y = e_x, e_y
        elif event == cv.EVENT_LBUTTONUP:
            w = e_x - x
            h = e_y - y
    cv2.imshow('1', image)
    cv2.setMouseCallback('1', mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x, y, w, h

def draw(res, pixel_size, x, y, w, h):
    block = res[y: y + h, x: x + w]
    for i in range(0, h, pixel_size):
        for j in range(0, w, pixel_size):
            color = np.mean(block[i: i + pixel_size, j: j + pixel_size], axis=(0, 1)).astype(np.uint8)
            block[i: i + pixel_size, j: j + pixel_size] = color

    res[y: y + h, x: x + w] = block
    return res

def pixelate(image, pixel_size):
    res = image.copy()
    x, y, w, h = get_area(res, pixel_size)
    print(x, y, w, h)
    return draw(res, pixel_size, x, y, w, h)

def sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    res = cv2.transform(image, sepia_filter)
    res = np.clip(res, 0, 255)

    return res

def main():
    args = cli_argument_parser()
    image = cv2.imread(args.image_path)
    result = None
    if args.task == 'idle':
        pass
    elif args.task == 'grayscale':
        result = gray_filter(image)
    elif args.task == 'sepia':
        result = sepia(image)
    elif args.task == 'resize':
        if (args.width < 0 or args.height < 0):
            raise ValueError('Unspecified or invalid width and height')
        result = resize(image, args.width, args.height)
    elif args.task == 'vignette':
        if args.radius < 0:
            raise ValueError('Unspecified or invalid radius')
        result = vignette(image, args.radius)
    elif args.task == 'pixelate':
        result = pixelate(image, args.pixel_size)
    else:
        raise ValueError('Unsupported mode')

    if args.task != 'idle':
        cv2.imwrite(args.out_image_path, result)

    cv2.imshow('original', image)
    if args.task != 'idle':
        cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)









