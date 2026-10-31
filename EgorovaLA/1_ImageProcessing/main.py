"""Применение фильтров к изображению.

    python main.py -i data/frog.jpg -f neon -p glow=1.5      один фильтр с параметрами
    python main.py -i data/frog.jpg -f antique fade matte    несколько фильтров
    python main.py -i data/frog.jpg -f all                   все фильтры
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Фильтры изображений")
    parser.add_argument("-i", "--input", required=True, help="входное изображение")
    parser.add_argument("-o", "--output", default="results",
                        help="файл (для одного фильтра) или папка, по умолчанию results")
    parser.add_argument("-f", "--filter", nargs="+", required=True,
                        help="фильтры: " + ", ".join(ImageFilter.names()) + " или all")
    parser.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE",
                        help="параметр фильтра (только для одного фильтра)")
    return parser.parse_args()


def parse_params(items):
    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"параметр '{item}' должен иметь вид KEY=VALUE")
        for cast in (int, float, str):
            try:
                params[key] = cast(value)
                break
            except ValueError:
                pass
    return params


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"файл не найден: {path}")
    # fromfile + imdecode работают с русскими путями в Windows
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"не удалось прочитать изображение: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"не удалось сохранить: {path}")
    buffer.tofile(path)


def main():
    # 1. аргументы
    args = cli_argument_parser()
    names = ImageFilter.names() if args.filter == ["all"] else args.filter

    # 2. чтение изображения
    try:
        image = read_image(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка чтения: {exc}", file=sys.stderr)
        return 1

    # 3. создание фильтров
    try:
        params = parse_params(args.param)
        if params and len(names) > 1:
            raise ValueError("параметры -p можно задать только для одного фильтра")
        filters = [ImageFilter.get_filter(name, **params) for name in names]
    except ValueError as exc:
        print(f"Ошибка фильтра: {exc}", file=sys.stderr)
        return 2

    # 4. применение и сохранение
    source = Path(args.input)
    output = Path(args.output)
    for name, image_filter in zip(names, filters):
        if len(names) == 1 and output.suffix:
            path = output
        else:
            path = output / f"{source.stem}_{name}{source.suffix}"
        try:
            save_image(path, image_filter.apply_filter(image))
        except (OSError, ValueError, cv2.error) as exc:
            print(f"Ошибка сохранения: {exc}", file=sys.stderr)
            return 3
        print(f"Готово: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
