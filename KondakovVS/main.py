import argparse
import os
import sys

import cv2
import numpy as np

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Применение фильтров к изображению (OpenCV + NumPy).")
    parser.add_argument("-i", "--image", required=True,
                        help="путь к входному изображению")
    parser.add_argument("-o", "--output", default=None,
                        help="путь для сохранения результата "
                             "(по умолчанию <имя>_<фильтр>.png рядом с исходным)")
    parser.add_argument("--show", action="store_true",
                        help="показать исходное и результирующее изображения")

    sub = parser.add_subparsers(dest="filter", required=True,
                                metavar="FILTER", help="тип фильтра")

    p = sub.add_parser("resize", help="изменение разрешения")
    p.add_argument("--width", type=int, help="новая ширина, px")
    p.add_argument("--height", type=int, help="новая высота, px")
    p.add_argument("--scale", type=float, help="коэффициент масштабирования")
    p.add_argument("--interpolation", choices=["nearest", "bilinear"],
                   default="bilinear")

    sub.add_parser("grayscale", help="перевод в оттенки серого")

    p = sub.add_parser("antique", help="фотоэффект «антиквариат» (сепия)")
    p.add_argument("--intensity", type=float, default=1.0, help="сила тонирования, [0, 1]")
    p.add_argument("--contrast", type=float, default=0.85, help="множитель контраста")

    p = sub.add_parser("fade", help="выцветание цветов")
    p.add_argument("--strength", type=float, default=0.6, help="сила эффекта, [0, 1]")

    p = sub.add_parser("film", help="имитация инфракрасной плёнки")
    p.add_argument("--mode", choices=["bw", "color"], default="bw",
                   help="bw — ч/б ИК-плёнка, color — цветная (Aerochrome)")
    p.add_argument("--halation", type=float, default=0.35, help="сила ореола, [0, 1]")
    p.add_argument("--grain", type=float, default=0.04, help="СКО зерна")
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("matte", help="овальная матовая рамка")
    p.add_argument("--radius", type=float, default=0.85, help="размер овала, доля полуосей")
    p.add_argument("--feather", type=float, default=0.25, help="ширина плавного перехода")
    p.add_argument("--color", default="#FFFFFF", help="цвет рамки #RRGGBB")

    p = sub.add_parser("old", help="состаренная фотография: царапины и шум")
    p.add_argument("--scratches", type=int, default=15, help="число царапин")
    p.add_argument("--dust", type=int, default=300, help="число пылинок")
    p.add_argument("--noise", type=float, default=0.06, help="СКО шума")
    p.add_argument("--sepia", type=float, default=0.8, help="сила сепии, [0, 1]")
    p.add_argument("--vignette", type=float, default=0.5, help="сила виньетки, [0, 1]")
    p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("neon", help="неоновый эффект")
    p.add_argument("--color", default="rainbow", help="'rainbow' или #RRGGBB")
    p.add_argument("--threshold", type=float, default=0.15, help="порог контуров, [0, 1)")
    p.add_argument("--glow", type=float, default=4.0, help="радиус свечения (sigma)")
    p.add_argument("--intensity", type=float, default=1.5, help="яркость свечения")
    p.add_argument("--background", type=float, default=0.15, help="яркость фона, [0, 1]")

    return parser


def read_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    if image.dtype != np.uint8:
        image = (image.astype(np.float32) / np.iinfo(image.dtype).max * 255).astype(np.uint8)
    return image


def write_image(path, image):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение в формат {ext}")
    buf.tofile(path)


def show_images(original, result):
    cv2.imshow("Original", original)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = cli_argument_parser().parse_args()
    params = {k: v for k, v in vars(args).items()
              if k not in ("image", "output", "show", "filter") and v is not None}

    try:
        image = read_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка чтения: {e}", file=sys.stderr)
        return 1

    try:
        image_filter = ImageFilter.get_filter(args.filter, **params)
        result = image_filter.apply_filter(image)
    except ValueError as e:
        print(f"Ошибка параметров фильтра: {e}", file=sys.stderr)
        return 2

    output = args.output
    if output is None:
        base = os.path.splitext(args.image)[0]
        output = f"{base}_{args.filter}.png"
    try:
        write_image(output, result)
    except (OSError, ValueError) as e:
        print(f"Ошибка сохранения: {e}", file=sys.stderr)
        return 3

    print(f"{args.filter}: {image.shape} -> {result.shape}, сохранено в {output}")
    if args.show:
        show_images(image, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
