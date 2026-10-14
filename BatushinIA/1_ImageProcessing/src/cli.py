# src/cli.py
import argparse


def cli_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Применение фильтров обработки изображений (OpenCV, PR1)."
    )
    parser.add_argument("-i", "--input",  required=True,  help="Путь к входному изображению")
    parser.add_argument("-o", "--output", required=True,  help="Путь для сохранения результата")
    parser.add_argument("-f", "--filter", required=True,
                        help="Имя фильтра: resize | gray | antique | fade | film | matte | scratches | neon")
    parser.add_argument("-W", "--width",  type=int, default=None, help="Ширина (для resize)")
    parser.add_argument("-H", "--height", type=int, default=None, help="Высота (для resize)")

    parser.add_argument("--vignette", type=float, default=0.3,
                    help="Сила виньетки для antique (0..1, по умолчанию 0.3)")
    
    parser.add_argument("--alpha", type=float, default=0.5,
                    help="Сохранение цвета для fade (0..1, по умолчанию 0.5)")
    parser.add_argument("--contrast", type=float, default=0.85,
                    help="Остаточный контраст для fade (0..1, по умолчанию 0.85)")
    parser.add_argument("--haze", type=float, default=0.15,
                    help="Дымка для fade (0..1, по умолчанию 0.15)")

    parser.add_argument("--grain", type=float, default=8.0,
                        help="Сила зерна (film, >=0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed ГПСЧ (film, для воспроизводимости)")

    parser.add_argument("--a_ratio", type=float, default=0.45,
                        help="Горизонтальная полуось (matte, 0..0.5)")
    parser.add_argument("--b_ratio", type=float, default=0.45,
                        help="Вертикальная полуось (matte, 0..0.5)")
    parser.add_argument("--feather", type=float, default=0.05,
                        help="Плавность границы (matte, >=0)")

    parser.add_argument("--n_scratches", type=int, default=5,
                        help="Количество царапин (scratches)")
    parser.add_argument("--scratch_intensity", type=float, default=40.0,
                        help="Яркость царапины (scratches)")
    parser.add_argument("--noise_level", type=float, default=12.0,
                        help="СКО шума (scratches)")

    parser.add_argument("--darkness", type=float, default=0.3,
                        help="Затемнение основы (neon, 0..1)")
    parser.add_argument("--neon_color", type=int, nargs=3, default=[255, 255, 0],
                        metavar=("B", "G", "R"),
                        help="Цвет неона в BGR (neon, каждое 0..255)")
    parser.add_argument("--intensity", type=float, default=1.0,
                        help="Сила свечения (neon)")

    args = parser.parse_args()

    if args.filter.lower() == "resize" and (args.width is None or args.height is None):
        parser.error("Для фильтра resize необходимо указать --width и --height")

    return args