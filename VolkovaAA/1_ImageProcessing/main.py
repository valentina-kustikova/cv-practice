import argparse
import sys
import logging
import cv2
from image_processing import ImageFilter
from utils import read_image


logging.basicConfig(level=logging.INFO, format='[ %(levelname)s ] %(message)s')
logger = logging.getLogger()


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='image',
                        type=str,
                        required=True,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='output',
                        default = "result.png",
                        type=str)
    parser.add_argument('-f', '--filter',
                        help='filter',
                        default = "grayscale",
                        type=str,
                        required=True)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--the_power_of_fading', type=float, default=0.25)
    parser.add_argument('--oval_min_r', type=float, default=0.65)
    parser.add_argument('--oval_max_r', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=40.0)
    parser.add_argument('--texture', type=str, default=argparse.SUPPRESS)

    args = parser.parse_args()
    return args


def main():
    args = cli_argument_parser()
    try:
        image = read_image(args.image_path)
        
        filter_processing = ImageFilter.get_filter(image, args.__dict__)
        result = filter_processing.apply_filter()
        cv2.imwrite(args.output, result)
    except Exception as ex:
        logger.error((f'{str(ex)}'))
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main() or 0)