# Скрипт запуска фильтров обработки изображений

import argparse
import os
import sys
import cv2

from filters.base import ImageFilter


def cli_argument_parser():
    # Разбор аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Обработка изображений с использованием OpenCV и numpy.'
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Путь к входному изображению')
    parser.add_argument('-o', '--output', required=True,
                        help='Путь для сохранения результата')
    parser.add_argument('-f', '--filter', required=True,
                        choices=['resize', 'gray', 'antique', 'fade',
                                 'film', 'matte', 'aged', 'neon'],
                        help='Тип фильтра')

    parser.add_argument('--scale', type=float, default=0.5,
                        help='Масштаб для resize (>0)')
    parser.add_argument('--vignette', type=float, default=0.4,
                        help='Сила виньетки в antique (>= 0)')
    parser.add_argument('--antique_noise', type=float, default=6.0,
                        help='Уровень зерна в antique (>= 0)')
    parser.add_argument('--strength', type=float, default=0.5,
                        help='Сила выцветания для fade (0..1)')
    parser.add_argument('--film_strength', type=float, default=0.7,
                        help='Сила эффекта плёнки (0..1)')
    parser.add_argument('--film_noise', type=float, default=8.0,
                        help='Уровень зерна для film')
    parser.add_argument('--size', type=float, default=0.9,
                        help='Размер маски matte (0..1)')
    parser.add_argument('--softness', type=float, default=0.15,
                        help='Мягкость края matte (>= 0)')
    parser.add_argument('--texture', type=str, default=None,
                        help='Путь к PNG-текстуре (обязателен для aged)')
    parser.add_argument('--noise_level', type=float, default=10.0,
                        help='Уровень шума для aged')
    parser.add_argument('--scratch_intensity', type=float, default=0.6,
                        help='Заметность текстуры в aged (0..1)')
    parser.add_argument('--neon_strength', type=float, default=9.0,
                        help='Усиление контура неона (>0)')
    parser.add_argument('--neon_threshold', type=float, default=20.0,
                        help='Порог контура неона (>=0): слабее — не рисуется')

    return parser.parse_args()


def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Файл не найден: {path}')
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f'Не удалось прочитать изображение: {path}')
    return image


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f'[ERROR] {e}')
        sys.exit(1)

    params = {}
    if args.filter == 'resize':
        params['scale'] = args.scale
    elif args.filter == 'antique':
        params['vignette_strength'] = args.vignette
        params['noise'] = args.antique_noise
    elif args.filter == 'fade':
        params['strength'] = args.strength
    elif args.filter == 'film':
        params['strength'] = args.film_strength
        params['noise_sigma'] = args.film_noise
    elif args.filter == 'matte':
        params['size'] = args.size
        params['softness'] = args.softness
    elif args.filter == 'aged':
        if args.texture is None:
            print('[ERROR] Для aged параметр --texture обязателен')
            sys.exit(1)
        params['texture_path'] = args.texture
        params['noise_level'] = args.noise_level
        params['scratch_intensity'] = args.scratch_intensity
    elif args.filter == 'neon':
        params['strength'] = args.neon_strength
        params['threshold'] = args.neon_threshold

    try:
        flt = ImageFilter.get_filter(args.filter, **params)
        result = flt.apply_filter(image)
    except Exception as e:
        print(f'[ERROR] Ошибка при применении фильтра: {e}')
        sys.exit(1)

    cv2.imwrite(args.output, result)
    print(f'[OK] Сохранено: {args.output}')


if __name__ == '__main__':
    main()