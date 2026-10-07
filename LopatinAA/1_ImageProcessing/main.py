import argparse
import sys

import cv2

from filters import ImageFilter

def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Использование фильтров для изображения"
    )
    parser.add_argument(
        "--filter", "-f", required=True,
        choices=["resize", "grayscale", "antique", "fadecolor", "tape", "matte", "noise", "neon"],
        help="Тип фильтра"
    )
    parser.add_argument("--input", "-i", required=True, help="Путь к входному изображению")
    parser.add_argument("--output", "-o", required=True, help="Путь к выходному изображению")

    # resize
    parser.add_argument("--width", type=int, default=None, help="Желаемая ширина (resize)")
    parser.add_argument("--height", type=int, default=None, help="Желаемая высота (resize)")

    # antique
    parser.add_argument("--intensity", type=float, default=0.8, help="Интенсивность эффекта (antique)")

    # fadecolor
    parser.add_argument("--strength", type=float, default=0.5, help="Степень выцветания (fadecolor)")

    # tape
    parser.add_argument("--grain", type=float, default=0.1, help="Зернистость (tape)")

    # matte
    parser.add_argument("--border", type=float, default=0.05, help="Ширина рамки (matte)")
    parser.add_argument("--feather", type=float, default=0.05, help="Cтепень закругления (matte)")

    # noise
    parser.add_argument("--scratch_count", type=int, default=7, help="Количество царапин (noise)")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Уровень шума (noise)")

    # neon
    parser.add_argument("--threshold", type=int, default=50, help="Порог выделения границ (neon)")
    parser.add_argument("--glow", type=float, default=0.8, help="Интенсивность подсветки (neon)")

    parser.add_argument("--color", type=int, nargs=3, default=[255, 255, 255], help="Установка цвета (b, r, g) (matte, neon)")

    return parser.parse_args()

def read_image(path):
    try:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
        return image
    except Exception as exc:
        print(f"Ошибка чтения изображения: {exc}", file=sys.stderr)
        sys.exit(1)

def main():
    args = cli_argument_parser()
    image = read_image(args.input)

    filter_type = args.filter.lower()
    params = {}

    if filter_type == 'resize':
        if args.width is not None:
            params['width'] = args.width
        if args.height is not None:
            params['height'] = args.height

    elif filter_type == 'antique':
        params['intensity'] = args.intensity

    elif filter_type == 'fadecolor':
        params['strength'] = args.strength

    elif filter_type == 'tape':
        params['grain'] = args.grain

    elif filter_type == 'matte':
        params['border'] = args.border
        params['feather'] = args.feather
        params['color'] = tuple(args.color)

    elif filter_type == 'noise':
        params['scratch_count'] = args.scratch_count
        params['noise_level'] = args.noise_level

    elif filter_type == 'neon':
        params['threshold'] = args.threshold
        params['glow'] = args.glow
        params['color'] = tuple(args.color)

    try:
        image_filter = ImageFilter.get_filter(filter_type, **params)
        result = image_filter.apply_filter(image)
    except Exception as exc:
        print(f'Ошибка применения фильтра: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        cv2.imwrite(args.output, result)
        print(f'Результат сохранён в {args.output}')
    except Exception as exc:
        print(f'Ошибка сохранения изображения: {exc}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
    