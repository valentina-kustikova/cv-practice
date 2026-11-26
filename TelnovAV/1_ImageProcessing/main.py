"""Точка входа в приложение обработки изображений."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from image_filters import ImageFilter


FILTER_NAMES = ["resize", "grayscale", "antique", "fade", "film", "matte", "scratches", "neon"]


def cli_argument_parser(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки с помощью argparse."""
    parser = argparse.ArgumentParser(
        description="Применение учебных фильтров к изображению с использованием OpenCV и NumPy."
    )
    parser.add_argument("--input", required=True, help="Путь к входному изображению")
    parser.add_argument("--output", required=True, help="Путь для сохранения результата")
    parser.add_argument(
        "--filter", required=True, choices=FILTER_NAMES,
        help="Фильтр: resize, grayscale, antique, fade, film, matte, scratches или neon",
    )
    parser.add_argument("--width", type=int, help="Новая ширина (для resize)")
    parser.add_argument("--height", type=int, help="Новая высота (для resize)")
    parser.add_argument(
        "--interpolation", choices=["nearest", "bilinear"], default="bilinear",
        help="Способ масштабирования: nearest или bilinear (по умолчанию bilinear)",
    )
    parser.add_argument(
        "--strength", type=float, default=0.8,
        help="Сила эффекта от 0 до 1 (для antique, fade, film, matte и neon)",
    )
    parser.add_argument(
        "--softness", type=float, default=0.28,
        help="Ширина плавного перехода для matte, число от 0.01 до значения меньше 1",
    )
    parser.add_argument(
        "--noise-strength", type=float, default=12.0,
        help="Сила шума для scratches (стандартное отклонение гауссова шума)",
    )
    parser.add_argument(
        "--scratch-count", type=int, default=40,
        help="Количество царапин для scratches",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Начальное значение генератора случайных чисел для scratches",
    )
    parser.add_argument(
        "--neon-color", choices=["cyan", "magenta", "green"], default="cyan",
        help="Цвет неонового свечения для neon",
    )
    return parser.parse_args(argv)


def read_image(filename: str | Path) -> np.ndarray:
    """Читает изображение и проверяет типичные ошибки ввода."""
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Указанный путь не является файлом: {path}")

    # IMREAD_COLOR даёт единый формат BGR для всех цветных фильтров.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Не удалось прочитать изображение '{path}'. Проверьте формат файла и права доступа."
        )
    if image.size == 0:
        raise ValueError(f"Изображение '{path}' пустое.")
    return image


def build_filter(args: argparse.Namespace) -> ImageFilter:
    """Подбирает параметры и создаёт выбранный фильтр через фабричный метод."""
    if args.filter == "resize":
        if args.width is None or args.height is None:
            raise ValueError("Для фильтра resize обязательно укажите --width и --height.")
        return ImageFilter.get_filter(
            "resize", width=args.width, height=args.height, interpolation=args.interpolation
        )
    if args.filter in {"antique", "fade", "film"}:
        return ImageFilter.get_filter(args.filter, strength=args.strength)
    if args.filter == "matte":
        return ImageFilter.get_filter(
            "matte", strength=args.strength, softness=args.softness
        )
    if args.filter == "scratches":
        return ImageFilter.get_filter(
            "scratches", noise_strength=args.noise_strength,
            scratch_count=args.scratch_count, seed=args.seed,
        )
    if args.filter == "neon":
        return ImageFilter.get_filter(
            "neon", strength=args.strength, color=args.neon_color
        )
    # Для grayscale дополнительные параметры не нужны.
    return ImageFilter.get_filter("grayscale")


def main(argv: list[str] | None = None) -> int:
    """Полный цикл: аргументы -> чтение -> фильтр -> запись результата."""
    try:
        args = cli_argument_parser(argv)
        image = read_image(args.input)
        image_filter = build_filter(args)
        result = image_filter.apply_filter(image)

        output_path = Path(args.output)
        # Если папка результата отсутствует, создаём её автоматически.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(output_path), result)
        if not saved:
            raise OSError(f"OpenCV не смог сохранить результат в файл: {output_path}")

        print(f"Фильтр '{args.filter}' применён успешно.")
        print(f"Размер исходного изображения: {image.shape[1]}x{image.shape[0]}")
        print(f"Размер результата: {result.shape[1]}x{result.shape[0]}")
        print(f"Результат сохранён: {output_path}")
        return 0
    except (FileNotFoundError, ValueError, OSError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
