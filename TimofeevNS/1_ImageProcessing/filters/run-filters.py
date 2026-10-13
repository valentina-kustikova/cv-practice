import argparse
import os
import random
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from image_filters import (
    add_white_noise,
    add_black_noise,
    add_salt_pepper_noise,
    reduce_intensity_at_point,
    apply_ellipse_mask,
    apply_film_defect,
    desaturate_region,
    apply_scratches,
    detect_edges,
    neonify,
    grow_region,
    otsu_binarize,
)


# ============================================================
# Реестр фильтров: имя -> функция, которая строит результат
# ============================================================
#
# Каждая функция принимает (image, ctx) и возвращает изображение.
# ctx — словарь с подготовленными данными (center, noise_count,
# film_region, desat_region, film_types), чтобы не считать одно
# и то же при каждом вызове.

def _f_white_noise(image, ctx):
    return add_white_noise(image, ctx['noise_count'])

def _f_black_noise(image, ctx):
    return add_black_noise(image, ctx['noise_count'])

def _f_salt_pepper(image, ctx):
    return add_salt_pepper_noise(image, ctx['noise_count'])

def _f_intensity_drop(image, ctx):
    return reduce_intensity_at_point(image, ctx['center'], factor=0.5)

def _f_ellipse_mask(image, ctx):
    return apply_ellipse_mask(image)

def _f_desaturate(image, ctx):
    return desaturate_region(image, ctx['desat_region'], percent=0.5)

def _f_scratches(image, ctx):
    return apply_scratches(image, scratch_coefficient=0.8,
                           scratch_color=(255, 255, 255), intensity=0.6,
                            max_segments = 10,
                            max_thickness = 3,
                            half_diag_fraction = 0.02)

def _f_edges(image, ctx):
    return detect_edges(image, thickness=2,
                        low_threshold=100, high_threshold=200)

def _f_neon(image, ctx):
    return neonify(image, thickness=3,
                   glow_color=(255, 0, 255), glow_intensity=0.8)

def _f_otsu(image, ctx):
    return otsu_binarize(image, invert=False)

def _f_otsu_inv(image, ctx):
    return otsu_binarize(image, invert=True)


# Фильмы плёнки динамически раскрываются в отдельные фильтры:
#   film_infrared, film_old_film, film_overexposed, film_faded, film_vintage
FILM_TYPES: Tuple[str, ...] = ('infrared', 'old_film', 'overexposed', 'faded', 'vintage')

def _make_film_filter(film_type: str) -> Callable:
    def _f(image, ctx):
        return apply_film_defect(image, film_type, ctx['film_region'])
    _f.__name__ = f'film_{film_type}'
    return _f


def build_filter_registry() -> Dict[str, Callable]:
    """Собирает словарь {имя_фильтра: функция}."""
    registry: Dict[str, Callable] = {
        'white_noise':  _f_white_noise,
        'black_noise':  _f_black_noise,
        'salt_pepper':  _f_salt_pepper,
        'intensity_drop': _f_intensity_drop,
        'ellipse_mask': _f_ellipse_mask,
        'desaturate':   _f_desaturate,
        'scratches':    _f_scratches,
        'edges':        _f_edges,
        'neon':         _f_neon,
        'otsu':         _f_otsu,
        'otsu_inv':     _f_otsu_inv,
    }
    for film_type in FILM_TYPES:
        registry[f'film_{film_type}'] = _make_film_filter(film_type)
    return registry


FILTERS: Dict[str, Callable] = build_filter_registry()


# ============================================================
# Вспомогательные функции
# ============================================================

def _prepare_context(image: np.ndarray, seed: int) -> dict:
    """Готовит всё, что нужно фильтрам: центр, шум, регионы."""
    random.seed(seed)
    np.random.seed(seed)

    h, w = image.shape[:2]
    center = (h // 2, w // 2)
    noise_count = max(1, (h * w) // 1000)

    return {
        'center': center,
        'noise_count': noise_count,
        'film_region': grow_region(image, center, depth=20),
        'desat_region': grow_region(image, center, depth=30),
    }


def _save(output_dir: str, base: str, filter_name: str, img: np.ndarray) -> str:
    """Сохраняет результат как <base>_<filter>.png. Возвращает путь."""
    path = os.path.join(output_dir, f'{base}_{filter_name}.png')
    cv2.imwrite(path, img)
    return path


def apply_filters(input_path: str,
                  filter_names: List[str],
                  output_dir: str = 'output',
                  seed: int = 42) -> List[str]:
    """
    Применяет указанные фильтры к изображению.

    Параметры
    ---------
    input_path : str
        Путь к входному изображению.
    filter_names : List[str]
        Имена фильтров из `FILTERS`. Если пустой список — применяются все.
    output_dir : str
        Каталог для результатов.
    seed : int
        Seed для воспроизводимости шумов/царапин.

    Возвращает список путей созданных файлов.
    """
    # Проверяем имена фильтров ДО чтения изображения
    unknown = [name for name in filter_names if name not in FILTERS]
    if unknown:
        raise ValueError(
            f'Неизвестный фильтр(ы): {", ".join(unknown)}.\n'
            f'Доступные: {", ".join(sorted(FILTERS.keys()))}'
        )

    if not filter_names:
        filter_names = list(FILTERS.keys())

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f'Не удалось прочитать: {input_path}')

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    h, w = image.shape[:2]
    print(f'Обрабатываю: {input_path} ({w}x{h})')
    print(f'Фильтры: {", ".join(filter_names)}')

    ctx = _prepare_context(image, seed)
    saved: List[str] = []

    for name in filter_names:
        func = FILTERS[name]
        result = func(image, ctx)
        path = _save(output_dir, base, name, result)
        saved.append(path)
        print(f'  -> {path}')

    return saved


# ============================================================
# CLI
# ============================================================

def cli_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Применяет фильтры обработки изображений.'
    )
    parser.add_argument(
        'input', nargs='?',
        help='Путь к входному изображению (не нужен при --list).'
    )
    parser.add_argument(
        '-f', '--filter', dest='filters', action='append', default=[],
        help='Имя фильтра (можно указать несколько раз: -f neon -f otsu). '
             'Если не указан — применяются все.'
    )
    parser.add_argument(
        '-o', '--output', default='output',
        help='Каталог для результатов (по умолчанию: output).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed для воспроизводимости (по умолчанию: 42).'
    )
    parser.add_argument(
        '-l', '--list', action='store_true',
        help='Вывести список доступных фильтров и выйти.'
    )
    return parser


def main() -> None:
    parser = cli_argument_parser()
    args = parser.parse_args()

    if args.list:
        print('Доступные фильтры:')
        for name in sorted(FILTERS.keys()):
            print(f'  {name}')
        return

    if args.input is None:
        parser.error('Не указан входной файл (или используйте --list).')

    try:
        saved = apply_filters(
            input_path=args.input,
            filter_names=args.filters,
            output_dir=args.output,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
        return

    print(f'\nГотово. Сохранено файлов: {len(saved)}')


if __name__ == '__main__':
    main()
