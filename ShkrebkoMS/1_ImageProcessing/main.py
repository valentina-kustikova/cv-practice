import argparse
import sys

import cv2
from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Применение фильтров к изображению")

    parser.add_argument("-i", "--input", required=True,
                        help="путь к входному изображению")
    parser.add_argument("-o", "--output", required=True,
                        help="путь для сохранения результата")
    parser.add_argument("-f", "--filter", required=True,
                        choices=["resize", "gray", "antique"],
                        help="тип фильтра")
    parser.add_argument("-p", "--param", type=float, default=None,
                        help="параметр фильтра (например, коэффициент масштабирования для resize)")

    return parser.parse_args()


def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
    return image


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
    except FileNotFoundError as error:
        print("Ошибка:", error)
        sys.exit(1)

    try:
        image_filter = ImageFilter.get_filter(args.filter, args.param)
    except ValueError as error:
        print("Ошибка:", error)
        sys.exit(1)

    result = image_filter.apply_filter(image)

    cv2.imwrite(args.output, result)
    print("Готово, результат сохранён в", args.output)


if __name__ == "__main__":
    main()