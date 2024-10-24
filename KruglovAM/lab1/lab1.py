import cv2 as cv
import fakeCv
import numpy as np
import argparse
import sys

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='output.jpg',
                        dest='out_image_path')
    parser.add_argument('-m', '--mode',
                        help='Mode (image, grey_color, resize, sepia_filter, vignette_filter, pixelate_region)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-w', '--width',
                        help='New width for resizing',
                        type=int,
                        default=200,
                        dest='width')
    parser.add_argument('-hg', '--height',
                        help='New height for resizing',
                        type=int,
                        default=200,
                        dest='height')
    parser.add_argument('-pxsz', '--pixel_size',
                        help='Size of pixels for pixelation',
                        type=int,
                        default=15,
                        dest='pixel_size')
    parser.add_argument('-r', '--radius',
                        help='Radius for vignette filter',
                        type=int,
                        default=100,
                        dest='radius')

    args = parser.parse_args()
    return args

def imgproc_samples(args):
    if args.image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    src_image = cv.imread(args.image_path)
    image_placeholder = src_image.copy()

    if(args.mode=="grey_color"):
        # Преобразование изображения в серое
        image_placeholder = fakeCv.bgr2grey(src_image)
        cv.imshow('My Gray image', image_placeholder)
        cv.waitKey(0)

    elif(args.mode=="resize"):
        # Изменение разрешения изображения
        image_placeholder = fakeCv.resize(src_image, args.width, args.height)
        cv.imshow('My resized image', image_placeholder)
        cv.waitKey(0)

    elif(args.mode=="sepia_filter"):
        # Функция применения сепии к изображению
        image_placeholder = fakeCv.apply_sepia(src_image)
        cv.imshow('My sepia image', image_placeholder)
        cv.waitKey(0)

    elif(args.mode=="vignette_filter"):
        # Функция применения фотоэффекта виньетки к изображению
        image_placeholder = fakeCv.apply_vignette(src_image, args.radius)
        cv.imshow('My vignette image', image_placeholder)
        cv.waitKey(0)

    elif(args.mode=="pixelate_region"):
        # Функция пикселизации региона изображения
        region = []
        cv.namedWindow("Choose the region to pixelate")
        cv.setMouseCallback("Choose the region to pixelate", fakeCv.select_region, [image_placeholder, region, args.pixel_size])
        while True:
            cv.imshow("Choose the region to pixelate", image_placeholder)
            key = cv.waitKey(1) & 0xFF

            if key == ord("r"):
                image_placeholder = src_image.copy()
                cv.setMouseCallback("Choose the region to pixelate", fakeCv.select_region, [image_placeholder, region, args.pixel_size])

            elif key == 13:  # Код клавиши Enter
                break
        cv.destroyWindow("Choose the region to pixelate")

    # Освобождение ресурсов для последующей работы
    cv.destroyAllWindows()
    cv.imwrite(args.out_image_path, image_placeholder)

def main():
    args = cli_argument_parser()
    
    imgproc_samples(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
