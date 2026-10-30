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
                        help='путь к входному изображению',
                        type=str,
                        required=True,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='путь для сохранения изображения',
                        type=str,
                        required=True,
                        dest='output_path')
    parser.add_argument('-f', '--filter',
                        help='название фильтра',
                        type=str,
                        required=True,
                        choices = ['resize', 'grayscale','antique', 'fadecolor',
                        'imitfilm', 'matte', 'noisescratches', 'neonlight'],
                        dest='filter_type')

    args = parser.parse_args()

    return args


def main():
    args = cli_argument_parser()
    try:
        image = read_image(args.image_path)
        filter_processing = ImageFilter.get_filter(image, args.filter_type)
        result = filter_processing.apply_filter()
        cv2.imwrite(args.output_path, result)
        logger.info('изображение сохранено успешно')
    except Exception as ex:
        logger.error((f'{str(ex)}'))


if __name__=='__main__':
    sys.exit(main() or 0)