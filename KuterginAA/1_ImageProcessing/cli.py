import argparse


def cli_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Приложение для применения фильтров к изображениям."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Путь для сохранения результата",
    )
    parser.add_argument(
        "-f",
        "--filter",
        required=True,
        type=str,
        choices=[
            "resize",
            "gray",
            "antique",
            "fade",
            "infrared",
            "matte",
            "old_photo",
            "neon",
        ],
        help="Тип применяемого фильтра",
    )

    # Дополнительные параметры
    parser.add_argument(
        "--width",
        type=int,
        default=300,
        help="Ширина для фильтра resize",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Высота для фильтра resize",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Коэффициент выцветания (fade)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="Уровень шума (old_photo)",
    )
    parser.add_argument(
        "--scratches",
        type=int,
        default=5,
        help="Количество царапин (old_photo)",
    )

    return parser.parse_args()