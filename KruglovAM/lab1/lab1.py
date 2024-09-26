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
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')
    parser.add_argument('-g', '--greyscale',
                    help='Convert image to greyscale',
                    type=bool,
                    default=False,
                    dest='gsc')
    parser.add_argument('-r', '--resize',
                    help='Resizes image',
                    type=bool,
                    default=False,
                    dest='rs')
    parser.add_argument('-s', '--sepia',
                    help='Apply sepia filter',
                    type=bool,
                    default=False,
                    dest='sep')
    parser.add_argument('-v', '--vignette',
                    help='Apply vignette filter',
                    type=bool,
                    default=False,
                    dest='vig')
    parser.add_argument('-p', '--pixelize',
                    help='Pixelize image',
                    type=bool,
                    default=False,
                    dest='pix')
    args = parser.parse_args()

    return args

def imgproc_samples(image_path, args):
    if image_path is None:
        raise ValueError('Empty path to the image')
    # Загрузка изображения
    src_image = cv.imread(image_path)

    if(args.gsc):
        # Преобразование изображения в серое
        my_gray_dst_image = fakeCv.bgr2grey(src_image)
        cv.imshow('My Gray image', my_gray_dst_image)
        cv.waitKey(0)

    # gray_dst_image = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray image', gray_dst_image)
    # cv.waitKey(0)

    if(args.rs):
        # Изменение разрешения изображения
        my_resized_image = fakeCv.resize(src_image, 100, 1000)
        cv.imshow('My resized image', my_resized_image)
        cv.waitKey(0)

    # resized_image = cv.resize(src_image, (100, 1000), interpolation=cv.INTER_AREA)
    # cv.imshow('Resized image', resized_image)
    # cv.waitKey(0)

    if(args.sep):
        # Функция применения фотоэффекта виньетки к изображению
        my_sepia_image = fakeCv.apply_sepia(src_image)
        cv.imshow('My sepia image', my_sepia_image)
        cv.waitKey(0)

    if(args.vig):
        # Функция применения фотоэффекта виньетки к изображению
        my_vignette_image = fakeCv.apply_vignette(src_image)
        cv.imshow('My vignette image', my_vignette_image)
        cv.waitKey(0)

    if(args.pix):
        # Функция применения фотоэффекта виньетки к изображению
        my_pixelate_image = fakeCv.pixelate_region(src_image, 0, 0, 275, 183, 20)
        cv.imshow('My pixelate image', my_pixelate_image)
        cv.waitKey(0)

    # Освобождение ресурсов для последующей работы
    cv.destroyAllWindows()

def main():
    args = cli_argument_parser()
    
    imgproc_samples(args.image_path, args)


if __name__ == '__main__':
    sys.exit(main() or 0)
