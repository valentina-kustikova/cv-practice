
import os
from abc import ABC, abstractmethod

from utils import list_image_files


class DataLoader(ABC):
    
    @property
    @abstractmethod
    def class_names(self):
        pass

    @abstractmethod
    def load(self, split):
        pass


class LandmarksDataLoader(DataLoader):
    """
    <root>/<split>/<source>/<NN_ClassName>/<image>
    """

    SPLITS = ("trainval", "test")
    SOURCES = ("NNSUDataset", "ExtDataset")

    def __init__(self, root, sources=SOURCES):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Директория не найдена: {root}")
        for source in sources:
            if source not in self.SOURCES:
                raise ValueError(f"Неизвестный источник данных: {source}. "
                                 f"Допустимые значения: {', '.join(self.SOURCES)}")
        self.root = root
        self.sources = list(sources)
        self._class_names = self._collect_class_names()

    def _collect_class_names(self):
        names = set()
        for split in self.SPLITS:
            for source in self.SOURCES:
                source_dir = os.path.join(self.root, split, source)
                if not os.path.isdir(source_dir):
                    continue
                for name in os.listdir(source_dir):
                    if os.path.isdir(os.path.join(source_dir, name)):
                        names.add(name)
        if len(names) == 0:
            raise ValueError(f"В {self.root} не найдено ни одного класса")
        return sorted(names)

    @property
    def class_names(self):
        return self._class_names

    def load(self, split):
        if split not in self.SPLITS:
            raise ValueError(f"Неизвестная часть набора данных: {split}")

        paths = []
        labels = []
        for source in self.sources:
            source_dir = os.path.join(self.root, split, source)
            if not os.path.isdir(source_dir):
                raise FileNotFoundError(f"Директория не найдена: {source_dir}")
            for label, class_name in enumerate(self._class_names):
                class_dir = os.path.join(source_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for path in list_image_files(class_dir):
                    paths.append(path)
                    labels.append(label)

        if len(paths) == 0:
            raise ValueError(f"В '{split}' не найдено ни одного изображения")
        return paths, labels
