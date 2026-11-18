import argparse
import sys
import logging
import cv2
from image_processing import ImageFilter


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger()

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--original',
                        help='path to the original image',
                        type=str,
                        required=True)
    parser.add_argument('-r', '--result',
                        help='path for saving the image',
                        type=str,
                        required=True)
    parser.add_argument('-f', '--filter',
                        help='which filter should I apply',
                        type=str,
                        required=True)
    parser.add_argument('--scale',
                        help='by what factor to compress the image',
                        type=float,
                        default=2,
                        required=False)
    parser.add_argument('--noise',
                        help='noise',
                        type=int,
                        default=12,
                        required=False)

    args = parser.parse_args()

    return args

def read_image(image_path):
    return cv2.imread(image_path)

def main():
    args = cli_argument_parser()
    try:
        image = read_image(args.original)
        filter_processing = ImageFilter.get_filter(image, args.__dict__)
        result = filter_processing.apply_filter()
        cv2.imwrite(args.result, result)
    except Exception as ex:
        logger.error((f'{str(ex)}'))


if __name__=='__main__':
    sys.exit(main() or 0)