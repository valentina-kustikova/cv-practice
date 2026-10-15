import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

import filters


FILTERS = {
    "resize": filters.resize,
    "grayscale": filters.grayscale,
    "antique": filters.antique,
    "fade": filters.fade,
    "film": filters.film,
    "matte": filters.matte,
    "scratches": filters.scratches,
    "neon": filters.neon,
}


def cli_argument_parser(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("-f", "--filter", choices=FILTERS)
    selected, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        description="Применение фильтра к изображению",
        epilog="Параметры фильтра: python main.py --filter ИМЯ --help",
    )
    parser.add_argument("-i", "--input", required=True, help="Входное изображение")
    parser.add_argument("-o", "--output", required=True, help="Файл результата")
    parser.add_argument("-f", "--filter", required=True, choices=FILTERS)
    if selected.filter == "resize":
        parser.add_argument("--width", type=int, required=True, help="Новая ширина > 0")
        parser.add_argument("--height", type=int, required=True, help="Новая высота > 0")
    elif selected.filter in ("antique", "fade"):
        default = 1.0 if selected.filter == "antique" else 0.5
        parser.add_argument("--strength", type=float, default=default,
                            help=f"Сила эффекта [0, 1], по умолчанию {default}")
    elif selected.filter == "film":
        parser.add_argument("--gamma", type=float, default=1.0,
                            help="Гамма негатива [0.1, 5], по умолчанию 1")
    elif selected.filter == "matte":
        parser.add_argument("--softness", type=float, default=0.3,
                            help="Ширина перехода [0.01, 1], по умолчанию 0.3")
    elif selected.filter == "scratches":
        parser.add_argument("--count", type=int, default=12,
                            help="Количество царапин >= 0, по умолчанию 12")
        parser.add_argument("--noise", type=float, default=10.0,
                            help="Стандартное отклонение шума [0, 255], по умолчанию 10")
        parser.add_argument("--seed", type=int, default=None,
                            help="Неотрицательное зерно генератора случайных чисел")
    elif selected.filter == "neon":
        parser.add_argument("--threshold", type=float, default=25.0,
                            help="Порог градиента [0, 255], по умолчанию 25")
        parser.add_argument("--radius", type=int, default=4,
                            help="Радиус свечения [0, 50], по умолчанию 4")
        parser.add_argument("--glow", type=float, default=2.0,
                            help="Яркость свечения [0, 10], по умолчанию 2")
    return parser.parse_args(argv)


def read_image(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Пустой файл: {filename}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось декодировать изображение: {filename}")
    return image


def save_image(filename, image):
    path = Path(filename)
    if not path.suffix:
        raise ValueError("Укажите расширение выходного файла, например .png или .jpg")
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"Не удалось закодировать изображение: {filename}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def main(argv=None):
    args = cli_argument_parser(argv)
    parameters = vars(args).copy()
    input_path = parameters.pop("input")
    output_path = parameters.pop("output")
    filter_name = parameters.pop("filter")
    try:
        image = read_image(input_path)
        result = FILTERS[filter_name](image, **parameters)
        save_image(output_path, result)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
