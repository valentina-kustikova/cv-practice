import argparse
import cv2
import sys
from image_filters import ImageFilter

def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Обработка изображений фильтрами OpenCV")

    parser.add_argument("--filter", "-f", required=True,
                        choices=["resize", "gray", "antique", "fade",
                                 "film", "matte", "scratches", "neon"],
                        help="Тип фильтра")
    parser.add_argument("--input", "-i", required=True,
                        help="Путь к входному изображению")
    parser.add_argument("--output", "-o", required=True,
                        help="Путь для сохранения результата")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Коэффициент масштаба (для resize)")
    parser.add_argument("--intensity", type=float, default=1.0,
                        help="Интенсивность эффекта (для antique/fade/neon)")
    parser.add_argument("--border", type=float, default=0.15,
                    help="Отступ овальной маски (для matte)")
    parser.add_argument("--threshold", type=float, default=50,
                        help="Порог выделения контуров (для neon)")

    return parser.parse_args()


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Ошибка: не удалось прочитать файл {path}")
        sys.exit(1)
    return image


def main():
    args = cli_argument_parser()  # разбор аргументов

    image = read_image(args.input)  # чтение изображения

    # Выбор фильтра по имени

    filter_obj = ImageFilter.get_filter(
        args.filter,
        scale=args.scale,
        intensity=args.intensity,
        border=args.border,
        threshold=args.threshold
    )
    
    result = filter_obj.apply_filter(image)  # применение

    cv2.imwrite(args.output, result)  # сохранение
    print(f"Готово: {args.output}")

if __name__ == "__main__":
    main()