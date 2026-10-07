import argparse
import sys

import numpy as np
from PIL import Image

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Применение фильтра к изображению')
    parser.add_argument('input', help='путь к входному RGB-изображению')
    parser.add_argument('output', help='путь для результата')
    parser.add_argument('--filter', required=True,
                        choices=['resize', 'gray', 'antique', 'fade', 'film',
                                 'matte', 'scratches', 'neon'])
    parser.add_argument('--new-h', type=int)
    parser.add_argument('--new-w', type=int)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--fade-color', type=int, nargs=3,
                        metavar=('R', 'G', 'B'), default=(200, 200, 200))
    parser.add_argument('--grain-sigma', type=float, default=10)
    parser.add_argument('--noise-sigma', type=float, default=15)
    parser.add_argument('--num-scratches', type=int, default=7)
    parser.add_argument('--sob-threshold', type=float, default=50)
    parser.add_argument('--seed', type=int)
    return parser.parse_args()


def read_image(path):
    try:
        with Image.open(path) as image:
            img = np.array(image)
    except FileNotFoundError:
        raise ValueError('Файл не найден: ' + path)
    except OSError:
        raise ValueError('Не удалось прочитать изображение: ' + path)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('Нужно RGB-изображение с тремя каналами')
    return img


def main():
    args = cli_argument_parser()
    if args.filter == 'resize' and (args.new_h is None or args.new_w is None):
        print('Для resize нужны --new-h и --new-w', file=sys.stderr)
        return 1

    try:
        img = read_image(args.input)
        params = vars(args).copy()
        params.pop('filter')
        params.pop('input')
        params.pop('output')
        image_filter = ImageFilter.get_filter(args.filter, **params)
        result = image_filter.apply_filter(img)
        Image.fromarray(result).save(args.output)
    except (ValueError, OSError) as error:
        print('Ошибка: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
