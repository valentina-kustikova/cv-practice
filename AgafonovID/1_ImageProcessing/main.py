import argparse
import os
import cv2
from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Практическая работа №1. Фильтры изображений OpenCV")
    parser.add_argument("--filter", "-f", type=str, required=True, help="Тип фильтра (grayscale, antique, resize, fade, film, matte, scratches, neon)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Путь к входному изображению")
    parser.add_argument("--output", "-o", type=str, required=True, help="Путь для сохранения результата")
    
    # Дополнительные параметры
    parser.add_argument("--width", type=int, help="Ширина для фильтра resize")
    parser.add_argument("--height", type=int, help="Высота для фильтра resize")
    parser.add_argument("--alpha", type=float, default=0.6, help="Коэффициент прозрачности для fade")
    
    return parser.parse_args()


def read_image(image_path: str):
    if not os.path.exists(image_path):
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
            "alpha": args.alpha
        }
        
        filter_obj = ImageFilter.get_filter(args.filter, **kwargs)
        result_image = filter_obj.apply_filter(image)
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        cv2.imwrite(args.output, result_image)
        print(f"Фильтр '{args.filter}' применен. Результат в '{args.output}'.")
        
    except Exception as e:
        print(f"Ошибка выполнения: {e}")


if __name__ == "__main__":
    main()