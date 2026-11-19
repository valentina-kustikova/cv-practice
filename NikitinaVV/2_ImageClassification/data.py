import logging
import os
from abc import ABC, abstractmethod
from collections import Counter

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
REDUCED_FLAGS = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2))


def read_image(path, min_side=None):
    """Читает цветное изображение (BGR)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Файл не найден: {path}')
    flags = cv2.IMREAD_COLOR
    if min_side:
        with Image.open(path) as header:
            side = min(header.size)
        flags = next((flag for factor, flag in REDUCED_FLAGS if side // factor >= min_side), flags)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Не удалось прочитать изображение: {path}')
    return image


class DataLoader(ABC):
    """Интерфейс загрузки размеченного набора данных. 
    Возвращает список пар (путь к изображению, имя класса)
    """
    def __init__(self, root):
        if not os.path.isdir(root):
            raise FileNotFoundError(f'Директория набора данных не найдена: {root}')
        self.root = root
        self.class_names = self.find_classes()
        if not self.class_names:
            raise ValueError(f'В наборе данных {root} не найдено ни одного класса')

    @abstractmethod
    def find_classes(self):
        """Упорядоченный список имен классов. Номер класса = индекс в этом списке."""

    @abstractmethod
    def list_samples(self, split):
        """Список пар (путь, имя класса) выборки split: 'train' или 'test'."""

    def load(self, split):
        """Пути к изображениям выборки и номера их классов."""
        samples = self.list_samples(split)
        if not samples:
            raise ValueError(f'Выборка {split} пуста')
        index = {name: i for i, name in enumerate(self.class_names)}
        unknown = {name for _, name in samples if name not in index}
        if unknown:
            raise ValueError(f'В выборке {split} есть классы, которых нет в обучающей: {sorted(unknown)}')
        paths = [path for path, _ in samples]
        labels = np.array([index[name] for _, name in samples])
        counts = Counter(name for _, name in samples)
        logger.info('Выборка %s: %d изображений, %d классов, от %d до %d на класс', split, len(paths),
                    len(counts), min(counts.values()), max(counts.values()))
        return paths, labels


def list_images(folder):
    return sorted(os.path.join(folder, name) for name in os.listdir(folder)
                  if name.lower().endswith(IMAGE_EXTENSIONS))


class LandmarksDataLoader(DataLoader):
    """Датасет фотографий достопримечательностей городов РФ."""
    SPLITS = {'train': 'trainval', 'test': 'test'}
    SOURCES = ('NNSUDataset', 'ExtDataset')

    def source_dirs(self, split):
        split_dir = os.path.join(self.root, self.SPLITS[split])
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f'Нет директории выборки {split}: {split_dir}')
        return [os.path.join(split_dir, s) for s in self.SOURCES if os.path.isdir(os.path.join(split_dir, s))]

    def find_classes(self):
        names = set()
        for source in self.source_dirs('train'):
            names.update(d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d)))
        return sorted(names)

    def list_samples(self, split):
        samples = []
        for source in self.source_dirs(split):
            for name in sorted(os.listdir(source)):
                folder = os.path.join(source, name)
                if os.path.isdir(folder):
                    samples += [(path, name) for path in list_images(folder)]
        return samples


DATA_LOADERS = {'landmarks': LandmarksDataLoader}
