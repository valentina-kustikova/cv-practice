"""Создаёт компактную копию датасета и записывает изменения в журнал."""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import shutil
import time

from PIL import Image

from dataset import ArchitectureDatasetLoader, read_image, sha256


MAX_SIDE = 640
JPEG_QUALITY = 85


def save_prepared_image(source_path, target_path):
    """Сохраняет RGB JPEG с прежними пропорциями и возвращает размеры до и после."""
    image = Image.fromarray(read_image(source_path))
    original_size = image.size
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(target_path, 'JPEG', quality=JPEG_QUALITY)
    return original_size, image.size


def copy_dataset_documents(source, destination):
    """Копирует описание датасета и таблицы источников с прежними путями."""
    for path in source.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.docx', '.csv'}:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def prepare(source, destination, make_zip=False):
    """Подготавливает копию датасета, записывает журнал и при необходимости создаёт ZIP."""
    started = time.perf_counter()
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Исходная папка и папка назначения должны быть отдельными и невложенными')
    if destination.exists():
        raise FileExistsError(f'Папка назначения уже существует: {destination}')
    archive = destination.with_name(destination.name + '.zip')
    if make_zip and archive.exists():
        raise FileExistsError(f'Архив уже существует: {archive}')

    loader = ArchitectureDatasetLoader(source)
    records = {split: loader.scan(split) for split in ('trainval', 'test')}
    hashes = {path: sha256(path) for images in records.values() for path, _ in images}
    test_hashes = {hashes[path] for path, _ in records['test']}

    destination.mkdir(parents=True)
    manifest = []
    excluded = []
    class_counts = {split: Counter() for split in records}
    prepared_bytes = 0
    for split, images in records.items():
        for path, label in images:
            relative_path = path.relative_to(source)
            original_hash = hashes[path]
            if split == 'trainval' and original_hash in test_hashes:
                excluded.append({
                    'original': relative_path.as_posix(),
                    'sha256': original_hash,
                    'reason': 'Точная копия изображения из test',
                })
                continue

            prepared_path = relative_path.with_name(relative_path.name + '.jpg')
            target = destination / prepared_path
            original_size, prepared_size = save_prepared_image(path, target)
            manifest.append({
                'original': relative_path.as_posix(),
                'prepared': prepared_path.as_posix(),
                'class': label,
                'original_sha256': original_hash,
                'prepared_sha256': sha256(target),
                'original_size': original_size,
                'prepared_size': prepared_size,
            })
            class_counts[split][label] += 1
            prepared_bytes += target.stat().st_size
            if len(manifest) % 250 == 0:
                logging.info('Подготовлено снимков: %d', len(manifest))

    for split, images in records.items():
        missing = {label for _, label in images} - class_counts[split].keys()
        if missing:
            raise ValueError(f'В {split} после подготовки не осталось снимков классов: {sorted(missing)}')
    copy_dataset_documents(source, destination)

    report = {
        'format_version': 1,
        'max_side': MAX_SIDE,
        'quality': JPEG_QUALITY,
        'reason_resize': 'Уменьшить размер файлов и ускорить чтение с сохранением пропорций.',
        'reason_duplicates': 'Исключить точные копии тестовых снимков из обучающей части.',
        'class_balance': 'Дополнительная балансировка классов не выполняется.',
        'excluded_trainval': excluded,
        'images': manifest,
        'counts': {split: dict(sorted(counts.items())) for split, counts in class_counts.items()},
        'original_bytes': sum(path.stat().st_size for path in source.rglob('*') if path.is_file()),
        'prepared_image_bytes': prepared_bytes,
        'seconds': time.perf_counter() - started,
    }
    (destination / 'preparation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    logging.info('Снимков: %d, исключено дубликатов: %d, JPEG: %.2f МиБ, время: %.2f с',
                 len(manifest), len(excluded), prepared_bytes / 2**20, report['seconds'])
    if make_zip:
        shutil.make_archive(str(destination), 'zip', destination.parent, destination.name)
        logging.info('Архив %s (%.2f МиБ), общее время: %.2f с',
                     archive, archive.stat().st_size / 2**20, time.perf_counter() - started)
    return report


def main():
    """Читает аргументы командной строки и запускает подготовку."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Папка исходного датасета')
    parser.add_argument('--destination', type=Path, required=True, help='Новая папка для копии')
    parser.add_argument('--zip', action='store_true', help='Создать ZIP рядом с подготовленной папкой')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        prepare(args.source, args.destination, args.zip)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Ошибка подготовки: {exc}\n')


if __name__ == '__main__':
    main()
