import argparse
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

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
    apply_sepia,
    apply_aging,
    # виньетки
    edxdy,
    edxdy_masks,
    vignette_rings,
    vignette_rings_masks,
    vignette_random_rings,
    vignette_random_rings_masks,
    vignette_smooth,
    _apply_vignette_rings_sets,
    _apply_vignette_rings_masks,
    _apply_vignette_random_sets,
    _apply_vignette_random_masks,
    Point,
    FILM_DEFECT_TABLE,
)


# ============================================================
# Реестр фильтров
# ============================================================

def _f_white_noise(image, ctx):
    return add_white_noise(image, ctx['noise_count'])

def _f_black_noise(image, ctx):
    return add_black_noise(image, ctx['noise_count'])

def _f_salt_pepper(image, ctx):
    return add_salt_pepper_noise(image, ctx['noise_count'])

def _f_intensity_drop(image, ctx):
    return np.clip(image.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)

def _f_ellipse_mask(image, ctx):
    return apply_ellipse_mask(image)

def _f_desaturate(image, ctx):
    f = image.astype(np.float32)
    gray = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
    gray_img = np.stack([gray, gray, gray], axis=-1)
    out = f + (gray_img - f) * 0.9
    return np.clip(out, 0, 255).astype(np.uint8)

def _f_scratches(image, ctx):
    return apply_scratches(image, scratch_coefficient=1.0,
                           scratch_color=(255, 255, 255), intensity=0.2,
                           max_segments=50,
                           max_thickness=3,
                           half_diag_fraction=0.02)

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

def _f_sepia(image, ctx):
    return apply_sepia(image, strength=ctx['sepia_strength'])

def _f_aging(image, ctx):
    return apply_aging(
        image,
        ctx['center'],
        sepia_strength=ctx['sepia_strength'],
        scratch_coefficient=1.0,
        scratch_color=(255, 255, 255),
        scratch_intensity=0.4,
        max_segments=50,
        max_thickness=2,
        half_diag_fraction=0.02,
        vignette_h_semi=ctx['vignette_h_semi'],
        vignette_v_semi=ctx['vignette_v_semi'],
        vignette_dx=ctx['vignette_dx'],
        vignette_dy=ctx['vignette_dy'],
        vignette_brighten=ctx['vignette_brighten'],
        vignette_interval_len=ctx['vignette_interval_len'],
        vignette_strength=ctx['vignette_strength'],
    )


# ---- Виньетки ----

def _f_vignette_rings(image, ctx):
    return vignette_rings(
        image, ctx['center'],
        h_semi=ctx['vignette_h_semi'],
        v_semi=ctx['vignette_v_semi'],
        dx=ctx['vignette_dx'],
        dy=ctx['vignette_dy'],
        brighten=ctx['vignette_brighten'],
        strength=ctx['vignette_strength'],
    )

def _f_vignette_rings_masks(image, ctx):
    return vignette_rings_masks(
        image, ctx['center'],
        h_semi=ctx['vignette_h_semi'],
        v_semi=ctx['vignette_v_semi'],
        dx=ctx['vignette_dx'],
        dy=ctx['vignette_dy'],
        brighten=ctx['vignette_brighten'],
        strength=ctx['vignette_strength'],
    )

def _f_vignette_random_rings(image, ctx):
    return vignette_random_rings(
        image, ctx['center'],
        h_semi=ctx['vignette_h_semi'],
        v_semi=ctx['vignette_v_semi'],
        dx=ctx['vignette_dx'],
        dy=ctx['vignette_dy'],
        brighten=ctx['vignette_brighten'],
        interval_len=ctx['vignette_interval_len'],
        strength=ctx['vignette_strength'],
    )

def _f_vignette_random_rings_masks(image, ctx):
    return vignette_random_rings_masks(
        image, ctx['center'],
        h_semi=ctx['vignette_h_semi'],
        v_semi=ctx['vignette_v_semi'],
        dx=ctx['vignette_dx'],
        dy=ctx['vignette_dy'],
        brighten=ctx['vignette_brighten'],
        interval_len=ctx['vignette_interval_len'],
        strength=ctx['vignette_strength'],
    )

def _f_vignette_smooth(image, ctx):
    return vignette_smooth(
        image,
        h_semi=ctx['vignette_h_semi'],
        v_semi=ctx['vignette_v_semi'],
        brighten=ctx['vignette_brighten'],
        strength=ctx['vignette_strength'],
    )



def _apply_film_whole(image: np.ndarray, film_type: str) -> np.ndarray:
    """Применяет дефект плёнки ко всему изображению (векторно)."""
    if film_type not in FILM_DEFECT_TABLE:
        return image.copy()

    spec = FILM_DEFECT_TABLE[film_type]
    kind = spec['defect']
    params = spec['params']
    f = image.astype(np.float32)

    if kind == 'darken':
        out = f * params['factor']
    elif kind == 'lighten':
        out = f + params['delta']
    elif kind == 'color_shift':
        out = f + np.array(params['shift'], dtype=np.float32)
    elif kind == 'desaturate':
        p = params['percent']
        gray = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
        gray_img = np.stack([gray, gray, gray], axis=-1)
        out = f + (gray_img - f) * p
    else:
        return image.copy()

    return np.clip(out, 0, 255).astype(np.uint8)

# ---- Фильмы плёнки ----

FILM_TYPES: Tuple[str, ...] = ('infrared', 'old_film', 'overexposed', 'faded', 'vintage')

def _make_film_filter(film_type: str) -> Callable:
    def _f(image, ctx):
        return _apply_film_whole(image, film_type)   # ← было: apply_film_defect(...)
    _f.__name__ = f'film_{film_type}'
    return _f


def build_filter_registry() -> Dict[str, Callable]:
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
        'sepia':        _f_sepia,
        'aging':        _f_aging,
        'vignette_rings':              _f_vignette_rings,
        'vignette_rings_masks':        _f_vignette_rings_masks,
        'vignette_random_rings':       _f_vignette_random_rings,
        'vignette_random_rings_masks': _f_vignette_random_rings_masks,
        'vignette_smooth':             _f_vignette_smooth,
    }
    for film_type in FILM_TYPES:
        registry[f'film_{film_type}'] = _make_film_filter(film_type)
    return registry


FILTERS: Dict[str, Callable] = build_filter_registry()


# ============================================================
# Вспомогательные функции
# ============================================================

def _prepare_context(image: np.ndarray,
                     seed: int,
                     vignette_params: Optional[dict] = None,
                     sepia_strength: float = 1.0) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    h, w = image.shape[:2]
    center = (h // 2, w // 2)
    noise_count = max(1, (h * w) // 50)

    vp = vignette_params or {}
    v_h_semi = vp.get('h_semi') if vp.get('h_semi') is not None else (w / 4.0)
    v_v_semi = vp.get('v_semi') if vp.get('v_semi') is not None else (h / 4.0)
    v_dx = vp.get('dx') if vp.get('dx') is not None else 10.0
    v_dy = vp.get('dy') if vp.get('dy') is not None else 10.0
    v_brighten = bool(vp.get('brighten', False))
    v_strength = float(vp.get('strength', 1.0))
    v_interval = float(vp.get('interval_len', 0.2))

    return {
        'center': center,
        'noise_count': noise_count,
        'film_region': grow_region(image, center, depth=20),
        'desat_region': grow_region(image, center, depth=30),
        'sepia_strength': float(sepia_strength),
        'vignette_h_semi': v_h_semi,
        'vignette_v_semi': v_v_semi,
        'vignette_dx': v_dx,
        'vignette_dy': v_dy,
        'vignette_brighten': v_brighten,
        'vignette_strength': v_strength,
        'vignette_interval_len': v_interval,
    }


def _save(output_dir: str, base: str, filter_name: str, img: np.ndarray) -> str:
    path = os.path.join(output_dir, f'{base}_{filter_name}.png')
    cv2.imwrite(path, img)
    return path


def apply_filters(input_path: str,
                  filter_names: List[str],
                  output_dir: str = 'output',
                  seed: int = 42,
                  vignette_params: Optional[dict] = None,
                  sepia_strength: float = 1.0) -> List[str]:
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

    ctx = _prepare_context(image, seed, vignette_params, sepia_strength)

    vignette_names = {
        'vignette_rings', 'vignette_rings_masks',
        'vignette_random_rings', 'vignette_random_rings_masks',
        'vignette_smooth', 'aging',
    }
    if any(n in vignette_names for n in filter_names):
        print('Параметры виньеток:')
        print(f"  center={ctx['center']}")
        print(f"  h_semi={ctx['vignette_h_semi']}, v_semi={ctx['vignette_v_semi']}")
        print(f"  dx={ctx['vignette_dx']}, dy={ctx['vignette_dy']}")
        print(f"  brighten={ctx['vignette_brighten']}, strength={ctx['vignette_strength']}")
        print(f"  interval_len={ctx['vignette_interval_len']}")

    if 'sepia' in filter_names or 'aging' in filter_names:
        print(f"  sepia_strength={ctx['sepia_strength']}")

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
    parser.add_argument('input', nargs='?',
                        help='Путь к входному изображению (не нужен при --list).')
    parser.add_argument('-f', '--filter', dest='filters', action='append', default=[],
                        help='Имя фильтра (можно несколько раз). Если не указан — все.')
    parser.add_argument('-o', '--output', default='output',
                        help='Каталог для результатов (по умолчанию: output).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed для воспроизводимости (по умолчанию: 42).')
    parser.add_argument('-l', '--list', action='store_true',
                        help='Вывести список доступных фильтров и выйти.')

    sepia = parser.add_argument_group('параметры сепии')
    sepia.add_argument('--sepia-strength', type=float, default=1.0,
                       dest='sepia_strength',
                       help='Сила сепии, 0..1 (по умолчанию 1.0).')

    vig = parser.add_argument_group('параметры виньеток')
    vig.add_argument('--vignette-h-semi', type=float, default=None,
                     dest='vignette_h_semi',
                     help='Полуось начального эллипса по X (по умолчанию w/4).')
    vig.add_argument('--vignette-v-semi', type=float, default=None,
                     dest='vignette_v_semi',
                     help='Полуось начального эллипса по Y (по умолчанию h/4).')
    vig.add_argument('--vignette-dx', type=float, default=10.0,
                     dest='vignette_dx',
                     help='Шаг роста по X на каждом кольце (по умолчанию 10).')
    vig.add_argument('--vignette-dy', type=float, default=10.0,
                     dest='vignette_dy',
                     help='Шаг роста по Y на каждом кольце (по умолчанию 10).')
    vig.add_argument('--vignette-mode', choices=['brighten', 'darken'],
                     default='darken', dest='vignette_mode',
                     help='Засветление или затемнение (по умолчанию darken).')
    vig.add_argument('--vignette-strength', type=float, default=1.0,
                     dest='vignette_strength',
                     help='Общая сила эффекта, 0..1 (по умолчанию 1.0).')
    vig.add_argument('--vignette-interval-len', type=float, default=0.2,
                     dest='vignette_interval_len',
                     help='Длина подинтервала для случайной виньетки (по умолчанию 0.2).')

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

    vignette_params = {
        'h_semi': args.vignette_h_semi,
        'v_semi': args.vignette_v_semi,
        'dx': args.vignette_dx,
        'dy': args.vignette_dy,
        'brighten': (args.vignette_mode == 'brighten'),
        'strength': args.vignette_strength,
        'interval_len': args.vignette_interval_len,
    }

    try:
        saved = apply_filters(
            input_path=args.input,
            filter_names=args.filters,
            output_dir=args.output,
            seed=args.seed,
            vignette_params=vignette_params,
            sepia_strength=args.sepia_strength,
        )
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
        return

    print(f'\nГотово. Сохранено файлов: {len(saved)}')


if __name__ == '__main__':
    main()
