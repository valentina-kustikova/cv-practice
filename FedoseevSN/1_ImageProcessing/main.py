import argparse
import os
import sys

import cv2

from filters import ImageFilter


def cli_argument_parser():
    # Здесь описаны все параметры, которые пользователь может передать
    # при запуске программы из командной строки.
    parser = argparse.ArgumentParser(description="Image processing")

    parser.add_argument("-i", "--input", required=True, help="Input image")
    parser.add_argument("-o", "--output", required=True, help="Output image")
    parser.add_argument(
        "-f", "--filter", required=True,
        choices=["resize", "gray", "antique", "fade", "film", "matte", "scratches", "neon"],
        help="Filter name",
    )

    # Настройки фильтров.
    parser.add_argument("--width", type=int, help="Width")
    parser.add_argument("--height", type=int, help="Height")
    parser.add_argument("--intensity", type=float, default=0.85)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--grain", type=float, default=15.0)
    parser.add_argument("--tint", type=float, default=0.4)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--softness", type=float, default=0.2)
    parser.add_argument("--noise-level", dest="noise_level", type=float, default=15.0)
    parser.add_argument("--scratch-count", dest="scratch_count", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=40.0)
    parser.add_argument("--glow-intensity", dest="glow", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=4.0, help="Glow blur radius")
    parser.add_argument("--vignette", type=float, default=0.55, help="Vignette intensity")

    return parser.parse_args()


def get_image(path):
    # OpenCV загружает цветное изображение в массив NumPy в порядке каналов BGR.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return img


def gather_kwargs(args):
    # Resize требует ширину и высоту.
    if args.filter == "resize":
        if not args.width or not args.height:
            raise ValueError("Specify --width and --height for resize")
        return {"width": args.width, "height": args.height}

    # Для каждого фильтра собираем только его собственные настройки.
    # Названия ключей должны совпадать с аргументами конструкторов классов.
    params = {
        "antique": {"intensity": args.intensity, "vignette": args.vignette},
        "fade": {"strength": args.strength},
        "film": {"grain": args.grain, "tint": args.tint},
        "matte": {"radius": args.radius, "softness": args.softness},
        "scratches": {
            "noise_level": args.noise_level,
            "scratch_count": args.scratch_count,
        },
        "neon": {"threshold": args.threshold, "glow": args.glow, "sigma": args.sigma},
    }
    return params.get(args.filter, {})


def main():
    # Сначала разбираем параметры запуска и получаем Namespace с их значениями.
    args = cli_argument_parser()

    try:
        # Загружаем исходное изображение и подготавливаем параметры фильтра.
        img = get_image(args.input)
        kwargs = gather_kwargs(args)

        # get_filter создаёт объект нужного класса, после чего этот объект
        # обрабатывает изображение своим методом apply_filter().
        img_filter = ImageFilter.get_filter(args.filter, **kwargs)
        processed = img_filter.apply_filter(img)

        # Если выходная папка ещё не существует, создаём её.
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # imwrite возвращает True при успешном сохранении и False при ошибке.
        if not cv2.imwrite(args.output, processed):
            raise IOError(f"Failed to write: {args.output}")

        print(f"Saved to {args.output}")

    except Exception as e:
        # Сюда попадают ошибки чтения, настройки, обработки и сохранения.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()