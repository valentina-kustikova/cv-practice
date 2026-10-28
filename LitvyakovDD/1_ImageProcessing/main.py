import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from inner.src import FILTERS, ImageFilter, check_image

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
            "  python main.py --image img.jpg --filter matte --matte-radius-x 0.8 --matte-radius-y 0.8 --matte-softness 0.2\n"
            "  python main.py --image img.jpg --filter scratches --strength 0.3\n"
            "  python main.py --image img.jpg --filter neon --neon-gain 9 --neon-brightness 0.4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", help="имя изображения в папке img")
    parser.add_argument("--filter", choices=FILTERS, help="название фильтра")
    parser.add_argument("--width", type=int, help="ширина результата для resize")
    parser.add_argument("--height", type=int, help="высота результата для resize")
    parser.add_argument("--strength", type=float, help="сила эффекта от 0 до 1")
    parser.add_argument("--matte-radius-x", type=float, help="радиус matte по X")
    parser.add_argument("--matte-radius-y", type=float, help="радиус matte по Y")
    parser.add_argument(
        "--matte-softness", type=float, help="мягкость перехода эффекта matte"
    )
    parser.add_argument("--neon-gain", type=float, help="усиление свечения neon")
    parser.add_argument(
        "--neon-brightness", type=float, help="яркость фона neon от 0 до 1"
    )
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

    if args.strength is not None and not 0 <= args.strength <= 1:
        parser.error("--strength должен быть от 0 до 1")
    if args.neon_gain is not None and args.neon_gain <= 0:
        parser.error("--neon-gain должен быть больше нуля")
    if args.neon_brightness is not None and not 0 <= args.neon_brightness <= 1:
        parser.error("--neon-brightness должен быть от 0 до 1")
    if args.matte_radius_x is not None and args.matte_radius_x <= 0:
        parser.error("--matte-radius-x должен быть больше нуля")
    if args.matte_radius_y is not None and args.matte_radius_y <= 0:
        parser.error("--matte-radius-y должен быть больше нуля")
    if args.matte_softness is not None and args.matte_softness < 0:
        parser.error("--matte-softness не может быть отрицательной")
    if args.strength is not None and args.filter not in (
        "antique", "fade", "film", "scratches"
    ):
        parser.error("--strength не используется этим фильтром")
    if args.filter != "matte" and (
        args.matte_radius_x is not None
        or args.matte_radius_y is not None
        or args.matte_softness is not None
    ):
        parser.error(
            "--matte-radius-x, --matte-radius-y и --matte-softness "
            "используются только с matte"
        )
    if args.filter != "neon" and (
        args.neon_gain is not None
        or args.neon_brightness is not None
    ):
        parser.error(
            "--neon-gain и --neon-brightness используются только с neon"
        )

    image_path = IMAGE_DIR / args.image
    try:
        image = read_image(image_path)
        check_image(image)

        parameters = {}
        if args.filter == "resize":
            parameters["width"] = args.width
            parameters["height"] = args.height
        if args.strength is not None:
            parameters["strength"] = args.strength
        if args.matte_radius_x is not None:
            parameters["matte_radius_x"] = args.matte_radius_x
        if args.matte_radius_y is not None:
            parameters["matte_radius_y"] = args.matte_radius_y
        if args.matte_softness is not None:
            parameters["matte_softness"] = args.matte_softness
        if args.neon_gain is not None:
            parameters["neon_gain"] = args.neon_gain
        if args.neon_brightness is not None:
            parameters["neon_brightness"] = args.neon_brightness
        result = ImageFilter.get_filter(args.filter, **parameters).apply_filter(image)
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
