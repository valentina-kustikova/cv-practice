import argparse
import cv2
import os
import sys

from numpy.ma.core import argsort

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Приложение для обработки изображений')
    parser.add_argument('-i', '--input', required=True, help='Путь к входному изображению')
    parser.add_argument('-o', '--output', required=True, help='Путь к выходному изображению')
    parser.add_argument('-f', '--filter', required=True, help='Тип фильтра (resize, gray, antique)')

    parser.add_argument('--scale', type=float, default=None, help='Масштаб для resize')
    parser.add_argument('--width', type=int, default=None, help='Ширина для resize')
    parser.add_argument('--height', type=int, default=None, help='Высота для resize')

    parser.add_argument('--vignette', type=float, default=0.6, help='Сила затемнения краев (0.0 - 1.0)')
    parser.add_argument('--vignette_scale', type=float, default=0.9, help='Размер светлого пятна (0.1 - 2.0)')
    parser.add_argument('--feather', type=float, default=0.3, help='Мягкость перехода (больше 0)')

    parser.add_argument('--contrast', type=float, default=0.6, help='Контраст для fade')
    parser.add_argument('--brightness', type=int, default=60, help='Яркость для fade')

    parser.add_argument('--margin', type=float, default=0.1, help='Размер рамки')

    parser.add_argument('--noise_level', type=int, default=20, help='Интенсивность шума')
    parser.add_argument('--scratches_count', type=int, default=15, help='Количество царапин')

    parser.add_argument('--thickness', type=int, default=2, help='Толщина неоновой границы')
    parser.add_argument('--blur_size', type=int, default=15, help='Степень блюра')
    return parser.parse_args()


def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Не удалось прочитать изображение ")
    return img


def main():
    args = cli_argument_parser()

    try:
        print(f"Чтение: {args.input}")
        img = read_image(args.input)

        print(f"Применение фильтра: {args.filter}")
        image_filter = ImageFilter.get_filter(
            args.filter,
            scale=args.scale,
            width=args.width,
            height=args.height,
            vignette = args.vignette,
            vignette_scale = args.vignette_scale,
            feather = args.feather,
            contrast = args.contrast,
            brightness = args.brightness,
            margin = args.margin,
            noise_level = args.noise_level,
            scratches_count = args.scratches_count,
            thickness=args.thickness,
            blur_size = args.blur_size,

        )

        result = image_filter.apply_filter(img)

        cv2.imwrite(args.output, result)
        print(f"Успешно сохранено: {args.output}")

    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()