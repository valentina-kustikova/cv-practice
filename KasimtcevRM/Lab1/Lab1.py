import argparse
import cv2 as cv
import sys
import numpy as np


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Input image path.',
                        required=True,
                        type=str,
                        dest='input')
    parser.add_argument('-o', '--output',
                        help='Output image path/name.',
                        type=str,
                        dest='output')
    parser.add_argument('-m', '--mode',
                        help='Image processing mode.',
                        required=True,
                        choices=['filter_greyscale', 'filter_resize', 'filter_sepia', 'filter_vignette', 'filter_pixelisation'],
                        type=str,
                        dest='mode')
    parser.add_argument('-p', '--param',
                        help='Parameters for the mode (comma-separated).',
                        type=str,
                        dest='param')
    args = parser.parse_args()

    return args

def get_param(param_string: str):
    return [float(item) for item in param_string.split(',')] 

def read_img(path):
    img = cv.imread(path)
    return cv.imread(path)

def whrite_ing(img, path):
    return cv.imwrite(path, img)

def filter_greyscale(image):
    _image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    _image[:,:] = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return _image

def filter_resize(image, _size_x, _size_y):
    size_x = int(_size_x)
    size_y = int(_size_y)
    _image = np.zeros((size_x, size_y, 3), dtype=np.uint8)
    iter_v = np.linspace(0, image.shape[0] - 1, size_y, dtype=int)
    iter_h = np.linspace(0, image.shape[1] - 1, size_x, dtype=int)
    iter_h, iter_v = np.meshgrid(iter_h, iter_v)
    _image = image[iter_v, iter_h]
    return _image

def filter_sepia(image, _intensity):
    _image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    filter_greyscale = filter_greyscale(image)
    _image[:,:,0] = np.clip(filter_greyscale - 1.0 * _intensity, 0, 255)
    _image[:,:,1] = np.clip(filter_greyscale + 0.5 * _intensity, 0, 255)
    _image[:,:,2] = np.clip(filter_greyscale + 2.0 * _intensity, 0, 255)
    return _image

def filter_vignette(image, _radius, _intensity):
    size_x, size_y = image.shape[1], image.shape[0]
    center_x, center_y = size_x // 2, size_y // 2
    x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
    center_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = np.ones((size_y, size_x), dtype=np.float32)
    mask[center_dist > _radius] = np.exp(-((center_dist[center_dist > _radius] - _radius)**2) * _intensity)
    _image = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    _image[:, :, 0] = np.uint8(image[:, :, 0] * mask)
    _image[:, :, 1] = np.uint8(image[:, :, 1] * mask)
    _image[:, :, 2] = np.uint8(image[:, :, 2] * mask)
    return _image

def filter_pixelisation(image, box, _block_size, _reset):
    _x1, _y1, _x2, _y2 = box
    block_size = int(_block_size)
    _image = image.copy()
    pixelized = _image[_y1:_y2, _x1:_x2]
    blocks_x = (_x2 - _x1) // block_size
    blocks_y = (_y2 - _y1) // block_size
    avg_color_list = np.zeros((blocks_y * blocks_x, 3), dtype=np.uint8)

    for i in range(blocks_y):
        for j in range(blocks_x):
            avg_color = np.mean(pixelized[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size], axis=(0, 1)).astype(np.uint8)
            avg_color_list[i * blocks_x + j] = avg_color

    if (bool(_reset)):
        np.random.shuffle(avg_color_list)

    for i in range(blocks_y):
        for j in range(blocks_x):
            pixelized[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = avg_color_list[i * blocks_x + j]

    _image[_y1:_y2, _x1:_x2] = pixelized
    return _image


def mouse_callback(event, x, y, flags, param):
    global x1, y1, x2, y2
    if event == cv.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if flags == cv.EVENT_FLAG_LBUTTON:
            x2, y2 = x, y
            _image = param.copy()
            cv.rectangle(_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imshow("Input image", _image)

def main():
    args = cli_argument_parser()
    input_img = read_img(args.input)
    output_img = None
    params_list = None
    if args.param:
        params_list = get_param(args.param)

    if args.mode == 'filter_greyscale':
        output_img = filter_greyscale(input_img)
    elif args.mode == 'filter_resize':
        output_img = filter_resize(input_img, params_list[0], params_list[1])
    elif args.mode == 'filter_sepia':
        output_img = filter_sepia(input_img, params_list[0])
    elif args.mode == 'filter_vignette':
        output_img = filter_vignette(input_img, params_list[0], params_list[1])
    else:
        cv.namedWindow("Input image")
        global x1, y1, x2, y2
        cv.setMouseCallback("Input image", mouse_callback, param=input_img)
        cv.imshow("Input image", input_img)
        cv.waitKey(0)

        output_img = filter_pixelisation(input_img, [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], params_list[0], params_list[1])

    cv.imshow('Input image', input_img)
    cv.imshow('Output image', output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if args.output:
        whrite_ing(output_img, args.output)
if (__name__ == '__main__'):
    sys.exit(main() or 0)
