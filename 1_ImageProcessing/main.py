import argparse
import sys
import os
import cv2
import numpy as np

from filters.base import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Применение фильтров к изображению (OpenCV, ПР №1)."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Путь к входному изображению")
    parser.add_argument("-o", "--output", required=True,
                        help="Путь для сохранения результата")
    parser.add_argument("-f", "--filter", required=True,
                        choices=["resize", "gray", "antique", "fade",
                                 "film", "matte", "scratches", "neon"],
                        help="Тип фильтра")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--brightness", type=int, default=25)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--vignette", type=float, default=0.6)
    parser.add_argument("--border", type=int, default=40)
    parser.add_argument("--n_scratches", type=int, default=25)
    parser.add_argument("--noise_std", type=float, default=10.0)
    parser.add_argument("--threshold", type=int, default=40)
    return parser.parse_args()


def read_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image


def save_image(path, image):
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f"Не удалось сохранить изображение: {path}")


def build_filter_kwargs(args):
    if args.filter == "resize":
        return {"scale": args.scale}
    if args.filter == "fade":
        return {"alpha": args.alpha, "brightness": args.brightness}
    if args.filter == "film":
        return {"strength": args.strength}
    if args.filter == "antique":
        return {"vignette": args.vignette}
    if args.filter == "matte":
        return {"border": args.border}
    if args.filter == "scratches":
        return {"n_scratches": args.n_scratches, "noise_std": args.noise_std}
    if args.filter == "neon":
        return {"threshold": args.threshold}
    return {}


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
        print(f"[+] Изображение загружено: shape={image.shape}")
    except Exception as e:
        print(f"[-] Ошибка чтения: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        image_filter = ImageFilter.get_filter(args.filter, **build_filter_kwargs(args))
        result = image_filter.apply_filter(image)
        print(f"[+] Фильтр '{args.filter}' применён")
    except Exception as e:
        print(f"[-] Ошибка применения фильтра: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        save_image(args.output, result)
        print(f"[+] Сохранено: {args.output}")
    except Exception as e:
        print(f"[-] Ошибка сохранения: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
