import argparse
import os
import sys
import tkinter

import cv2
import numpy as np

from filters import ImageFilter

FILTERS = ['resize', 'gray', 'antique', 'fade', 'film', 'matte', 'old', 'neon']


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Применение фильтров к изображению') #создание объекта парсера с описанием программы
    parser.add_argument('-i', '--input', required=True, help='путь к входному изображению')
    parser.add_argument('-o', '--output', help='путь для сохранения (по умолчанию <имя>_<фильтр>.png)')
    parser.add_argument('-f', '--filter', required=True, choices=FILTERS, help='тип фильтра')
    parser.add_argument('--show', action='store_true', help='показать исходное и полученное изображения')

    params = parser.add_argument_group('параметры фильтров')
    params.add_argument('--width', type=int, help='resize: новая ширина')
    params.add_argument('--height', type=int, help='resize: новая высота')
    params.add_argument('--scale', type=float, help='resize: коэффициент масштабирования')
    params.add_argument('--strength', type=float, default=0.5, help='fade: сила эффекта от 0 до 1')
    params.add_argument('--grain', type=float, default=12, help='film: сила зерна')
    params.add_argument('--radius', type=float, default=0.7, help='matte: радиус овала')
    params.add_argument('--softness', type=float, default=0.3, help='matte: ширина перехода к белому')
    params.add_argument('--noise', type=float, default=20, help='old: сила шума')
    params.add_argument('--scratches', type=int, default=30, help='old: количество царапин')
    params.add_argument('--threshold', type=float, default=0.2, help='neon: порог контуров от 0 до 1')
    params.add_argument('--color', type=int, nargs=3, default=[255, 0, 255], metavar=('R', 'G', 'B'),
                        help='neon: цвет свечения')
    params.add_argument('--seed', type=int, help='film, old: зерно генератора случайных чисел')
    return parser.parse_args() 


def read_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Файл не найден: {path}')
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) 
        raise ValueError(f'Не удалось прочитать изображение: {path}')
    return image


def save_image(path, image):
    ok, buffer = cv2.imencode(os.path.splitext(path)[1], image)
    if not ok:
        raise ValueError(f'Не удалось сохранить изображение: {path}')
    buffer.tofile(path)


def show_fullscreen(title, image):
    root = tkinter.Tk()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight() 
    root.destroy() 
    h, w = image.shape[:2] 
    scale = min(screen_w / w, screen_h / h) 
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) 
    top, left = (screen_h - new_h) // 2, (screen_w - new_w) // 2 
    canvas = cv2.copyMakeBorder(resized, top, screen_h - new_h - top, left, screen_w - new_w - left,
                                cv2.BORDER_CONSTANT) 
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(title, canvas)
    cv2.waitKey(0)


def main():
    args = cli_argument_parser()
    try:
        image = read_image(args.input)
        result = ImageFilter.get_filter(args).apply_filter(image) 
        name = os.path.splitext(os.path.basename(args.input))[0]
        output = args.output or f'{name}_{args.filter}.png'
        save_image(output, result)
        print(f'Результат сохранен в {output}')

        if args.show:
            show_fullscreen('Input', image)
            show_fullscreen('Result', result)
            cv2.destroyAllWindows()
    except (OSError, ValueError, cv2.error) as e:
        print(f'Ошибка: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
