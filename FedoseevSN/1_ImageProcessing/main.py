import argparse
import os
import sys

import cv2

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Обработка изображений")
    parser.add_argument("-i", "--input", required=True, help="Путь к входному изображению")
    parser.add_argument("-o", "--output", required=True, help="Путь для результата")
    parser.add_argument(
        "-f", "--filter", required=True,
        choices=["resize", "gray", "antique", "fade", "film", "matte", "scratches", "neon"],
        help="Тип фильтра",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.3)
    parser.add_argument("--grain", type=float, default=15.0)
    parser.add_argument("--tint", type=float, default=0.6)
    parser.add_argument("--border", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=0.05)
    parser.add_argument("--n-scratches", dest="n_scratches", type=int, default=15)
    parser.add_argument("--noise-level", dest="noise_level", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--glow-intensity", dest="glow_intensity", type=float, default=0.8)
    return parser.parse_args()


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def build_params(args):
    if args.filter == "resize":
        if args.width is None or args.height is None:
            raise ValueError("Для фильтра resize укажите --width и --height")
        return {"width": args.width, "height": args.height}
    if args.filter == "antique":
        return {"intensity": args.intensity}
    if args.filter == "fade":
        return {"strength": args.strength}
    if args.filter == "film":
        return {"grain": args.grain, "tint": args.tint}
    if args.filter == "matte":
        return {"border": args.border, "softness": args.softness}
    if args.filter == "scratches":
        return {"n_scratches": args.n_scratches, "noise_level": args.noise_level}
    if args.filter == "neon":
        return {"threshold": args.threshold, "glow_intensity": args.glow_intensity}
    return {}


def main():
    args = cli_argument_parser()
    try:
        image = read_image(args.input)
        params = build_params(args)
        image_filter = ImageFilter.get_filter(args.filter, **params)
        result = image_filter.apply_filter(image)

        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        if not cv2.imwrite(args.output, result):
            raise IOError(f"Не удалось сохранить изображение: {args.output}")

        print(f"Готово: {args.output}")
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()