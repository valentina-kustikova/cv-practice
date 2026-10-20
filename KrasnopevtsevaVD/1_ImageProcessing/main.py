import argparse
import cv2
import sys
import os
from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Применение фильтров к изображениям.')
    parser.add_argument('-i', '--input', required=True, help='Путь к входному изображению')
    parser.add_argument('-o', '--output', required=True, help='Путь для сохранения результата')
    parser.add_argument(
        '-f', '--filter', required=True,
        choices=['resize', 'grayscale', 'antique', 'fade', 'infrared', 'matte', 'aged', 'neon'],
        help='Тип фильтра'
    )
    parser.add_argument('--scale', type=float, default=0.5, help='Коэффициент масштабирования')
    parser.add_argument('--alpha', type=float, default=0.5, help='Прозрачность')
    return parser.parse_args()


def read_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение {file_path}.")
    return img


def save_image(file_path, img):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not cv2.imwrite(file_path, img):
        raise IOError(f"Не удалось сохранить изображение {file_path}.")


def main():
    args = cli_argument_parser()

    try:
        img = read_image(args.input)
        filter_params = {}
        if args.filter == 'resize':
            filter_params['scale'] = args.scale
        elif args.filter == 'fade':
            filter_params['alpha'] = args.alpha
        image_filter = ImageFilter.get_filter(args.filter, **filter_params)
        result_img = image_filter.apply_filter(img)

        save_image(args.output, result_img)

    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()