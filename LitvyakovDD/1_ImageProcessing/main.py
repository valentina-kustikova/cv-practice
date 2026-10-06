import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from inner.src import FILTERS, check_image

IMAGE_DIR = Path(__file__).parent / "img"
RESULT_DIR = Path(__file__).parent / "result"


def read_image(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось открыть изображение.")
    return image


def write_image(filename, image):
    extension = filename.suffix
    success, data = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Не удалось сохранить изображение.")
    filename.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(filename)

def make_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        epilog=(
            "Examples:\n"
            "  python main.py --image img.jpg --filter grayscale\n"
            "  python main.py --image img.jpg --filter resize --width 800 --height 600\n"
            "  python main.py --image img.jpg --filter scratches --count 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", help="имя изображения в папке img")
    parser.add_argument("--filter", choices=FILTERS, help="название фильтра")
    parser.add_argument("--width", type=int, help="ширина результата для resize")
    parser.add_argument("--height", type=int, help="высота результата для resize")
    parser.add_argument("--count", type=int, help="количество царапин для scratches")
    parser.add_argument("--strength", type=float, help="сила эффекта от 0 до 1")
    parser.add_argument("--threshold", type=float, help="порог контуров для neon")
    parser.add_argument("--brightness", type=float, help="яркость фона neon от 0 до 1")
    return parser


def main(argv=None):
    parser = make_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    if not args.image or not args.filter:
        parser.error("нужно указать --image и --filter")

    if Path(args.image).name != args.image:
        parser.error("--image должен быть именем файла из папки img")

    if args.filter == "resize":
        if args.width is None or args.height is None:
            parser.error("для resize укажите --width и --height")
        if args.width <= 0 or args.height <= 0:
            parser.error("--width и --height должны быть больше нуля")
    elif args.width is not None or args.height is not None:
        parser.error("--width и --height используются только с resize")

    if args.filter == "scratches":
        if args.count is None:
            parser.error("для scratches укажите --count")
        if args.count < 0:
            parser.error("--count не может быть отрицательным")
    elif args.count is not None:
        parser.error("--count используется только с scratches")

    if args.strength is not None and not 0 <= args.strength <= 1:
        parser.error("--strength должен быть от 0 до 1")
    if args.threshold is not None and args.threshold < 0:
        parser.error("--threshold не может быть отрицательным")
    if args.brightness is not None and not 0 <= args.brightness <= 1:
        parser.error("--brightness должен быть от 0 до 1")
    if args.strength is not None and args.filter not in (
        "antique", "fade", "film", "matte", "neon"
    ):
        parser.error("--strength не используется этим фильтром")
    if args.filter != "neon" and (
        args.threshold is not None or args.brightness is not None
    ):
        parser.error("--threshold и --brightness используются только с neon")

    image_path = IMAGE_DIR / args.image
    try:
        image = read_image(image_path)
        check_image(image)

        parameters = {}
        if args.filter == "resize":
            parameters["width"] = args.width
            parameters["height"] = args.height
        elif args.filter == "scratches":
            parameters["count"] = args.count
        if args.strength is not None:
            parameters["strength"] = args.strength
        if args.threshold is not None:
            parameters["threshold"] = args.threshold
        if args.brightness is not None:
            parameters["brightness"] = args.brightness

        result = FILTERS[args.filter](image, **parameters)
        filter_number = list(FILTERS).index(args.filter) + 1
        output_path = RESULT_DIR / (
            f"{image_path.stem}_{filter_number}{image_path.suffix}"
        )
        write_image(output_path, result)
    except (OSError, ValueError, TypeError, cv2.error) as error:
        parser.error(str(error))

    print("Результат сохранён:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
