from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import logging
import cv2
import numpy as np

LOG = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    source: str

class DataLoader(ABC):
    """Абстрактный интерфейс загрузчика датасета."""

    @abstractmethod
    def load(self, split: str) -> Iterator[ImageRecord]:
        """Загружает только запрошенную часть датасета."""
        raise NotImplementedError

    def iter_split(self, split: str) -> Iterator[ImageRecord]:
        # Обратная совместимость со старым интерфейсом.
        return iter(self.load(split))

    @abstractmethod
    def classes(self) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def read_image(path: str | Path) -> np.ndarray | None:
        """Читает изображение, в том числе по путям с кириллицей в Windows."""
        path = Path(path)
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                LOG.warning("Не удалось декодировать изображение: %s", path)
            return image
        except (OSError, ValueError) as exc:
            LOG.warning("Не удалось прочитать %s: %s", path, exc)
            return None

class LandmarkDataLoader(DataLoader):
    """
    Загрузчик структуры:
      DATASET/
        trainval/{NNSUDataset,ExtDataset}/ИМЯ_КЛАССА/*.jpg
        test/{NNSUDataset,ExtDataset}/ИМЯ_КЛАССА/*.jpg
    Каталоги источников необязательны: допускается и DATASET/trainval/КЛАСС/*.jpg.
    """

    SPLITS = ("trainval", "test")
    SOURCES = ("NNSUDataset", "ExtDataset")

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Директория датасета не найдена: {self.root}")
        if not (self.root / "trainval").is_dir() or not (self.root / "test").is_dir():
            raise FileNotFoundError(
                "Ожидаются директории trainval/ и test/ внутри корня датасета."
            )
        # Классы определяются исключительно по trainval. test не сканируется
        # при создании загрузчика, чтобы не использовать тестовую выборку в обучении.
        self._classes = self._discover_classes(self.root / "trainval")

    def _class_dirs(self, split_dir: Path):
        for source in self.SOURCES:
            source_dir = split_dir / source
            if source_dir.is_dir():
                for class_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                    yield class_dir
        # Поддержка варианта без промежуточных каталогов источников.
        for class_dir in sorted(p for p in split_dir.iterdir()
                                if p.is_dir() and p.name not in self.SOURCES):
            yield class_dir

    def _discover_classes(self, trainval_dir: Path) -> list[str]:
        names = {class_dir.name.strip() for class_dir in self._class_dirs(trainval_dir)}
        if not names:
            raise ValueError("В trainval/ не найдены каталоги классов.")
        return sorted(names)

    def classes(self) -> list[str]:
        return list(self._classes)

    def load(self, split: str) -> Iterator[ImageRecord]:
        if split not in self.SPLITS:
            raise ValueError(f"Неизвестная выборка {split!r}; допустимо: {self.SPLITS}")
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Не найдена выборка: {split_dir}")
        for class_dir in self._class_dirs(split_dir):
            source = class_dir.parent.name if class_dir.parent.name in self.SOURCES else "unspecified"
            label = class_dir.name.strip()
            if split == "test" and label not in self._classes:
                LOG.warning("Класс из test отсутствует в trainval и будет исключён: %s", label)
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield ImageRecord(path=path, label=label, source=source)

    def read_refs(self) -> dict[str, str]:
        """Читает refs.csv из ExtDataset, если файл присутствует."""
        candidates = [
            self.root / "trainval" / "ExtDataset" / "refs.csv",
            self.root / "trainval" / "ExtDataset" / "<refs>.csv",
            self.root / "refs.csv",
            self.root / "<refs>.csv",
        ]
        refs = {}
        for csv_path in candidates:
            if not csv_path.is_file():
                continue
            try:
                import csv
                with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.reader(f):
                        if not row:
                            continue
                        # Формат: dir_name\image_name;web_resource
                        joined = ",".join(row)
                        if ";" in joined:
                            image_key, resource = joined.split(";", 1)
                        elif "," in joined:
                            image_key, resource = joined.split(",", 1)
                        else:
                            continue
                        refs[image_key.strip().replace("\\\\", "/")] = resource.strip()
                LOG.info("Прочитано %d ссылок из %s", len(refs), csv_path)
                break
            except (OSError, UnicodeError) as exc:
                LOG.warning("Не удалось прочитать %s: %s", csv_path, exc)
        return refs
