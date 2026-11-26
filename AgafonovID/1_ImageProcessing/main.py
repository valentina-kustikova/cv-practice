import argparse
import os
import sys
import cv2
from filters import ImageFilter

def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Обработка изображений с использованием библиотеки OpenCV")
    parser.add_argument("--filter", "-f", type=str, required=True, help="Тип фильтра (grayscale, antique, resize, fade, film, matte, scratches, neon)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Путь к входному изображению")
    parser.add_argument("--output", "-o", type=str, required=True, help="Путь для сохранения результата")
    
    parser.add_argument("--width", type=int, help="Ширина для фильтра resize")
    parser.add_argument("--height", type=int, help="Высота для фильтра resize")
    parser.add_argument("--alpha", type=float, default=0.6, help="Коэффициент прозрачности для fade (0..1)")
    parser.add_argument("--radius", type=float, default=0.85, help="Радиус овальной маски для matte")
    parser.add_argument("--noise-ratio", type=float, default=0.01, dest="noise_ratio", help="Доля шумовых пикселей для фильтра scratches")
    parser.add_argument("--num-lines", type=int, default=6, dest="num_lines", help="Количество царапин для фильтра scratches")
    return parser.parse_args()


def read_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Ошибка чтении файла: {image_path}")
    return image


def main():
    args = cli_argument_parser()
    
    try:
        image = read_image(args.input)
        
        kwargs = {
            "width": args.width,
            "height": args.height,
            "alpha": args.alpha,
            "radius": args.radius,
            "noise_ratio": args.noise_ratio,
            "num_lines": args.num_lines
        }
        
        filter_obj = ImageFilter.get_filter(args.filter, **kwargs)
        result_image = filter_obj.apply_filter(image)

        # сохраняем новое изображение
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if not cv2.imwrite(args.output, result_image):
            raise IOError(f"Не удалось сохранить изображение: {args.output}")

        print(f"Фильтр '{args.filter}' применен. Результат в '{args.output}'.")

    except Exception as e:
        print(f"Ошибка выполнения: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()