from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from image_filters import (
    ImageFilter,
    _parse_rgb,
    manual_resize,
    read_image,
    save_image,
)


FILTERS = (
    "resize",
    "grayscale",
    "antique",
    "fade",
    "film",
    "matte",
    "noise",
    "neon",
)


def cli_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Практическая работа №1: базовая матричная обработка изображений."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Путь к входному изображению."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Путь к результирующему изображению."
    )
    parser.add_argument(
        "-f", "--filter",
        required=True,
        choices=FILTERS,
        help="Фильтр: resize, grayscale, antique, fade, film, matte, noise или neon."
    )

    # Resize
    parser.add_argument("--width", type=int, default=900, help="Ширина для resize.")
    parser.add_argument("--height", type=int, default=600, help="Высота для resize.")
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear"),
        default="bilinear",
        help="Метод интерполяции при resize."
    )

    # Antique
    parser.add_argument(
        "--antique-strength",
        type=float,
        default=1.0,
        help="Сила эффекта Antique, 0..1."
    )

    # Fade
    parser.add_argument(
        "--fade-alpha",
        type=float,
        default=0.35,
        help="Сила выцветания, 0..1."
    )
    parser.add_argument(
        "--fade-color",
        type=str,
        default="235,215,175",
        help="Цвет выцветания в RGB, например 255,220,180."
    )

    # Film
    parser.add_argument("--film-gamma", type=float, default=0.85)
    parser.add_argument("--film-grain", type=float, default=0.06)

    # Matte
    parser.add_argument("--matte-radius", type=float, default=0.58)
    parser.add_argument("--matte-strength", type=float, default=1.0)
    parser.add_argument("--matte-softness", type=float, default=2.2)

    # Scratches/noise
    parser.add_argument("--noise-amount", type=float, default=0.10)
    parser.add_argument("--scratch-density", type=float, default=0.03)

    # Neon
    parser.add_argument("--neon-strength", type=float, default=1.5)
    parser.add_argument("--neon-blur", type=float, default=0.8)

    return parser


def run(args: argparse.Namespace) -> None:
    image = read_image(args.input)
    fade_color = _parse_rgb(args.fade_color)

    if args.filter == "resize":
        result = manual_resize(
            image,
            width=args.width,
            height=args.height,
            interpolation=args.interpolation,
        )

    elif args.filter == "grayscale":
        result = ImageFilter.get_filter("grayscale").apply_filter(image)

    elif args.filter == "antique":
        result = ImageFilter.get_filter(
            "antique", strength=args.antique_strength
        ).apply_filter(image)

    elif args.filter == "fade":
        result = ImageFilter.get_filter(
            "fade",
            alpha=args.fade_alpha,
            color=fade_color,
        ).apply_filter(image)

    elif args.filter == "film":
        result = ImageFilter.get_filter(
            "film",
            gamma=args.film_gamma,
            grain=args.film_grain,
        ).apply_filter(image)

    elif args.filter == "matte":
        result = ImageFilter.get_filter(
            "matte",
            radius=args.matte_radius,
            strength=args.matte_strength,
            softness=args.matte_softness,
        ).apply_filter(image)

    elif args.filter == "noise":
        result = ImageFilter.get_filter(
            "noise",
            noise=args.noise_amount,
            scratch_density=args.scratch_density,
        ).apply_filter(image)

    elif args.filter == "neon":
        result = ImageFilter.get_filter(
            "neon",
            strength=args.neon_strength,
            blur=args.neon_blur,
        ).apply_filter(image)

    else:
        raise ValueError(f"Неподдерживаемый фильтр: {args.filter}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(str(output), result)
    print(f"Готово: {output.resolve()}")


def main() -> None:
    parser = cli_argument_parser()
    args = parser.parse_args()

    try:
        run(args)
    except (FileNotFoundError, IOError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
