"""Читает фотографии и собирает пути к ним вместе с метками классов."""
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
import hashlib
import logging
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def read_image(path):
    """Читает изображение как RGB-массив с учётом ориентации и прозрачности."""
    with Image.open(Path(path)) as image:
        image = ImageOps.exif_transpose(image)
        if 'A' in image.getbands() or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            image = Image.alpha_composite(background, rgba)
        return np.asarray(image.convert('RGB')).copy()


def sha256(path):
    """Вычисляет SHA-256 содержимого файла для поиска точных дубликатов."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class DataLoader(ABC):
    """Задаёт общий интерфейс для чтения путей к снимкам и их меток."""

    @abstractmethod
    def load(self, split, max_images_per_class=0, seed=42):
        """Возвращает список пар (путь к изображению, название класса)."""


class ArchitectureDatasetLoader(DataLoader):
    """Собирает снимки архитектурных объектов по папкам классов."""

    def __init__(self, root):
        self.root = Path(root)

    def scan(self, split):
        """Находит изображения и их классы в указанной части датасета."""
        if split not in {'trainval', 'test'}:
            raise ValueError('Часть датасета должна быть trainval или test')
        folder = self.root / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        records = [(p, p.parent.name) for p in sorted(folder.glob('*/*/*'))
                   if p.is_file() and p.suffix.lower() in EXTENSIONS]
        if not records:
            raise ValueError(f'Изображения не найдены в {folder}/source/class')
        return records

    def load(self, split, max_images_per_class=0, seed=42):
        """Возвращает записи выборки, исключив дубликаты и ограничив число изображений."""
        if max_images_per_class < 0:
            raise ValueError('Максимальное число изображений на класс не может быть отрицательным')
        records = self.scan(split)
        if split == 'trainval':
            test_hashes = {sha256(p) for p, _ in self.scan('test')}
            filtered = [(p, c) for p, c in records if sha256(p) not in test_hashes]
            logging.info('Исключено точных дубликатов из обучения: %d', len(records)-len(filtered))
            if {c for _, c in records} != {c for _, c in filtered}:
                raise ValueError('После удаления дубликатов один из классов остался без обучающих изображений')
            records = filtered
        if max_images_per_class:
            rng = np.random.default_rng(seed)
            selected = []
            for label in sorted({c for _, c in records}):
                group = [r for r in records if r[1] == label]
                indices = rng.choice(len(group), min(len(group), max_images_per_class), replace=False)
                selected.extend(group[i] for i in sorted(indices))
            records = selected
        logging.info('%s: снимков — %d, классов — %d. Снимков по классам: %s', split, len(records),
                     len({c for _, c in records}), dict(sorted(Counter(c for _, c in records).items())))
        return records
