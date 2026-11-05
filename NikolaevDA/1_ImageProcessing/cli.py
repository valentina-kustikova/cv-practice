import argparse
import cv2
import sys
from img_filter import ImageFilter

def read_image(src):
    image = cv2.imread(src, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {src}")
    return image

def save_image(path, image):
    if not cv2.imwrite(path, image):
        raise OSError(f"Не удалось сохранить изображение: {path}")

FILTERS = ['resize', 'gray', 'antique', 'fade', 'film', 'matte', 'old', 'neon']

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, required=True, help="input a path to the image")
    parser.add_argument("-o", "--output", type=str, required=True, help="input a path to where the image is gonna be saved")
    parser.add_argument('-f', '--filter', type=str, required=True, choices=FILTERS, help='filter type')

    parser.add_argument("--width", type=int, help="width of the resized image")
    parser.add_argument("--height", type=int, help="height of the resized image")
    parser.add_argument("--fade", type=float, default=0.5, help="the strength of the fade of the image")
    parser.add_argument("--grain", type=float, default=12, help="the strength of the grain on the image")
    parser.add_argument("--seed", type=int, help="seed of the rng")
    parser.add_argument("--radius", type=float, default=0.7, help="radius of the matte filter")
    parser.add_argument("--softness", type=float, default=0.3, help="softness of the matte filter")
    parser.add_argument("--intensity", type=float, default=30.0, help="intensity of old photo filter")
    parser.add_argument("--scratches", type=int, default=30, help="amount of scratches on the old photo")

    return parser.parse_args()

def main():
    args = cli_argument_parser()
    try: 
        image = read_image(args.input)
        result = ImageFilter.get_filter(args).apply_filter(image)
        save_image(args.output, result)
    except (OSError, ValueError, cv2.error) as e:
        print(f'Ошибка: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()