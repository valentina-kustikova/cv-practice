"""
run_filters.py

Применяет все фильтры из image_filters.py к изображению и сохраняет каждый
результат в отдельный файл.
"""

import os
import random

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
    all_points,
    reduce_intensity,
)


def apply_all_filters(input_path: str,
                      output_dir: str = 'output',
                      seed: int = 42) -> list:
    """
    Применяет все фильтры к изображению `input_path`.
    Каждый результат сохраняется в `output_dir/<base>_<filter>.png`.

    Возвращает список созданных файлов.
    """
    random.seed(seed)
    np.random.seed(seed)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f'Не удалось прочитать: {input_path}')

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    h, w = image.shape[:2]
    center = (h // 2, w // 2)

    # Сколько шумовых точек добавить
    noise_count = max(1, (h * w) // 30)

    # Регионы для фильтров, работающих по области
    film_region = all_points(image) # grow_region(image, center, depth=20)
    desat_region = all_points(image) # grow_region(image, center, depth=30)

    saved: list = []

    def save(name: str, img: np.ndarray) -> None:
        path = os.path.join(output_dir, f'{base}_{name}.png')
        cv2.imwrite(path, img)
        saved.append(path)
        print(f'  -> {path}')

    print(f'Обрабатываю: {input_path} ({w}x{h})')

    # 1. Белый шум
    save('white_noise', add_white_noise(image, noise_count))

    # 2. Чёрный шум
    save('black_noise', add_black_noise(image, noise_count))

    # 3. Соль и перец
    save('salt_pepper', add_salt_pepper_noise(image, noise_count))

    # 4. Понижение интенсивности в центральной точке
    save('intensity_drop',
         reduce_intensity(image, all_points(image), factor=0.5)) # переделать на цикл

    # 6. Эллиптическая маска
    save('ellipse_mask', apply_ellipse_mask(image))

    # 8. Дефекты плёнки (все типы из таблицы)
    for film_type in ('infrared', 'old_film', 'overexposed', 'faded', 'vintage'):
        save(f'film_{film_type}',
             apply_film_defect(image, film_type, film_region))

    # 10. Засеривание области вокруг центра
    save('desaturate',
         desaturate_region(image, desat_region, percent=0.5))

    # 12. Царапины
    save('scratches',
         apply_scratches(image, scratch_coefficient=1.0,
                         scratch_color=(255, 255, 255), intensity=0.8))

    # 13. Края (сохранятся как ч/б маска)
    save('edges', detect_edges(image, thickness=2,
                               low_threshold=100, high_threshold=200))

    # 14. Неон
    save('neon', neonify(image, thickness=3,
                         glow_color=(255, 0, 255), glow_intensity=0.8))

    return saved


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Применить все фильтры к изображению')
    parser.add_argument('input', help='Путь к входному изображению')
    parser.add_argument('-o', '--output', default='output',
                        help='Каталог для результатов (по умолчанию: output)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed для воспроизводимости (default:42)')
    args = parser.parse_args()

    files = apply_all_filters(args.input, args.output, args.seed)
    print(f'\nГотово. Сохранено файлов: {len(files)}')
