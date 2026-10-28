"""Загрузчики наборов данных: список пар (путь к изображению, индекс класса)."""

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DataLoader(ABC):
    """Интерфейс загрузки размеченного набора данных.

    Наследник отвечает за формат хранения разметки и возвращает выборку
    как список (path, label_index). Изображения не читаются заранее, чтобы
    не держать весь набор данных в памяти.
    """

    def __init__(self, root, limit_per_class=None):
        self.root = Path(root)
        self.limit_per_class = limit_per_class
        if not self.root.exists():
            raise FileNotFoundError(f"Каталог с данными не найден: {self.root}")

    @property
    @abstractmethod
    def class_names(self):
        """Упорядоченный список имен классов; индекс в списке – метка класса."""

    @abstractmethod
    def _load_split(self, split):
        """Возвращает список (path, class_name) для выборки split ('train' или 'test')."""

    def get_samples(self, split):
        """Возвращает список (path, label_index), не более limit_per_class на класс."""
        index = {name: i for i, name in enumerate(self.class_names)}
        counts = {}
        samples = []
        for path, name in self._load_split(split):
            if self.limit_per_class is not None and counts.get(name, 0) >= self.limit_per_class:
                continue
            counts[name] = counts.get(name, 0) + 1
            samples.append((Path(path), index[name]))
        logger.info("Выборка '%s': %d изображений, по классам: %s", split, len(samples), counts)
        return samples


def _list_images(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


class IntelImageDataLoader(DataLoader):
    """Intel Image Classification (Kaggle: puneet6060/intel-image-classification).

    Ожидаемая структура архива (папка на каждый класс):
        root/seg_train/seg_train/{buildings,forest,glacier,mountain,sea,street}/*.jpg
        root/seg_test/seg_test/{...}/*.jpg
        root/seg_pred/seg_pred/*.jpg   (без разметки, удобно для режима predict)
    Повторяющийся вложенный каталог (seg_train/seg_train) необязателен.
    """

    SPLIT_DIRS = {"train": "seg_train", "test": "seg_test"}

    def __init__(self, root, limit_per_class=None):
        super().__init__(root, limit_per_class)
        self._class_names = sorted(p.name for p in self._split_dir("train").iterdir() if p.is_dir())
        if not self._class_names:
            raise ValueError(f"В {self._split_dir('train')} не найдено папок классов")

    def _split_dir(self, split):
        directory = self.root / self.SPLIT_DIRS[split]
        nested = directory / self.SPLIT_DIRS[split]
        if nested.is_dir():
            directory = nested
        if not directory.is_dir():
            raise FileNotFoundError(f"Каталог выборки '{split}' не найден: {directory}")
        return directory

    @property
    def class_names(self):
        return self._class_names

    def _load_split(self, split):
        split_dir = self._split_dir(split)
        for name in self._class_names:
            class_dir = split_dir / name
            if not class_dir.is_dir():
                logger.warning("Класс '%s' отсутствует в выборке '%s'", name, split)
                continue
            for path in _list_images(class_dir):
                yield path, name


class CsvDataLoader(DataLoader):
    """Набор данных с разметкой в CSV-файлах root/train.csv и root/test.csv.

    Каждая строка: path,label (путь относительно root). Пример другого формата
    хранения разметки, поддерживаемого тем же интерфейсом.
    """

    def __init__(self, root, limit_per_class=None):
        super().__init__(root, limit_per_class)
        self._class_names = sorted({name for _, name in self._read_csv("train")})

    def _read_csv(self, split):
        with open(self.root / f"{split}.csv", newline="", encoding="utf-8") as f:
            return [(self.root / row["path"], row["label"]) for row in csv.DictReader(f)]

    @property
    def class_names(self):
        return self._class_names

    def _load_split(self, split):
        return self._read_csv(split)


DATA_LOADERS = {
    "intel": IntelImageDataLoader,
    "csv": CsvDataLoader,
}
