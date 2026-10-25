import argparse


def cli_argument_parser():
    """Создаёт и обрабатывает аргументы командной строки."""

    parser = argparse.ArgumentParser(
        description="Библиотека фильтров обработки изображений"
    )

    # Пути к изображениям
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Путь к исходному изображению"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.jpg",
        help="Путь для сохранения результата"
    )

    # Выбор фильтра
    parser.add_argument(
        "-f", "--filter",
        required=True,
        choices=[
            "resize",
            "grayscale",
            "antique",
            "fade",
            "film",
            "matte",
            "noise",
            "neon",
        ],
        help="Название применяемого фильтра"
    )

    # Параметры изменения размера
    parser.add_argument("--width", type=int, default=None, help="Ширина результата")
    parser.add_argument("--height", type=int, default=None, help="Высота результата")
    parser.add_argument("--scale-x", type=float, default=0.5, help="Масштаб по ширине")
    parser.add_argument("--scale-y", type=float, default=0.5, help="Масштаб по высоте")

    # Параметры эффекта выцветания
    parser.add_argument("--alpha", type=float, default=0.35, help="Сила выцветания от 0 до 1")
    parser.add_argument(
        "--fade-color",
        default="235,215,175",
        help="Цвет выцветания в формате R,G,B"
    )

    # Параметры плёночного эффекта
    parser.add_argument("--strength", type=float, default=0.7, help="Сила плёночного эффекта")
    parser.add_argument("--film-noise", type=float, default=8.0, help="Шум плёночного эффекта")

    # Параметры эффекта старой фотографии
    parser.add_argument("--noise", type=float, default=0.1, help="Количество шума")
    parser.add_argument(
        "--scratch-density",
        type=float,
        default=0.1,
        help="Плотность царапин"
    )

    # Параметры матового эффекта
    parser.add_argument("--radius", type=float, default=0.58, help="Радиус светлой области")
    parser.add_argument("--matte-strength", type=float, default=1.0, help="Сила матового эффекта")
    parser.add_argument("--softness", type=float, default=2.2, help="Мягкость границ")

    # Параметры неонового эффекта
    parser.add_argument("--neon-strength", type=float, default=2.0, help="Яркость неонового свечения")
    parser.add_argument("--neon-blur", type=float, default=1.3, help="Размытие неонового свечения")

    return parser.parse_args()

