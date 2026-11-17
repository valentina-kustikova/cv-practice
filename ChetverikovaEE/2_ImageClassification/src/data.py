"""Загрузчик данных в виде дерева каталогов.

Интерфейс DataLoader и его реализация FolderDataLoader для набора данных,
хранящегося в виде дерева папок (файлы набора по ссылке распаковываются
вручную, загрузчику передаётся путь к распакованному каталогу).

Поддерживаются структуры выборок любой глубины вложенности:

    <root>/trainval/<имя_класса>/*.jpg                      -- плоская
    <root>/trainval/<поднабор>/<имя_класса>/*.jpg           -- вложенная
    <root>/trainval/<поднабор>/<имя_класса>/.../*.jpg       -- любая глубина

Например, если внутри trainval/ и test/ лежат два поднабора (ExtDataset,
NNSUDataset), а классы находятся внутри них, классами считаются самые
вложенные каталоги с изображениями; сами поднаборы классами не являются.

Правило определения классов: классом считается каталог, непосредственно
содержащий файлы изображений; именем класса — имя этого каталога.
Одноимённые классы из разных поднаборов объединяются в один класс, поэтому
одни и те же классы можно хранить в нескольких поднаборах одновременно.

Единый контракт: ``load(split) -> список пар (путь, метка класса)``,
где split — ``'trainval'`` (обучающая выборка) или ``'test'`` (тестовая).
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

Sample = Tuple[Union[str, Path], int]
"""Образец выборки: (путь к файлу изображения, номер класса)."""

VALID_SPLITS = ("trainval", "test")


class DataLoader(ABC):
    """Абстрактный интерфейс загрузчика данных, хранящихся в разных форматах.

    Изображение возвращается как путь к файлу (str/Path). Хранение путей
    позволяет не держать весь набор данных в памяти: изображения читаются
    по одному в момент извлечения признаков (ленивое чтение).
    """

    @abstractmethod
    def load(self, split: str) -> List[Sample]:
        """Загрузить выборку: split='trainval' — обучающая, 'test' — тестовая."""

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Имена классов; индекс имени совпадает с меткой класса."""

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Неизвестная выборка: '{split}'. Ожидалось одно из: {VALID_SPLITS}")


class FolderDataLoader(DataLoader):
    """Загрузчик набора данных в виде дерева каталогов (плоская и вложенная структура).

    Ожидаемые структуры (и любая их комбинация)::

        <root>/trainval/<имя_класса>/*.jpg
        <root>/trainval/<поднабор>/<имя_класса>/*.jpg
        <root>/test/...

    Особенности:

    * имена папок выборок распознаются автоматически (trainval/train,
      test/val/valid, без учёта регистра);
    * класс — каталог, непосредственно содержащий файлы изображений;
      классы ищутся рекурсивно, поэтому вложенность поднаборов любой глубины
      не мешает (поднабор — просто промежуточный каталог без изображений);
    * одноимённые классы из разных поднаборов объединяются в один класс;
    * метки классов назначаются по алфавитно отсортированному списку имён
      классов — это гарантирует одинаковые метки для trainval и test;
    * ``max_per_class`` ограничивает число изображений на класс суммарно
      по всем поднаборам.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    SPLIT_DIR_CANDIDATES = {"trainval": ("trainval", "train"), "test": ("test", "val", "valid")}

    def __init__(self, root: Union[str, Path], max_per_class: int = 0) -> None:
        """
        :param root: корень набора данных (каталог с папками trainval/ и test/)
        :param max_per_class: использовать не более N изображений на класс (0 — все)
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Каталог набора данных не найден: {self.root.resolve()}. "
                f"Передайте путь к каталогу, внутри которого лежат папки trainval/ и test/."
            )
        self.split_dirs = {split: self._resolve_split_dir(split) for split in VALID_SPLITS}
        self.max_per_class = int(max_per_class)
        # классы определяются по обучающей выборке; тестовая проверяется на совпадение в load()
        self._classes = self._discover_class_names(self.split_dirs["trainval"])
        logger.info("Набор данных: %s; классы (%d): %s", self.root, len(self._classes), ", ".join(self._classes))

    # ------------------------------------------------------------------
    # Поиск структуры каталогов
    # ------------------------------------------------------------------

    def _resolve_split_dir(self, split: str) -> Path:
        """Найти папку выборки в корне набора данных (без учёта регистра имени)."""
        candidates = self.SPLIT_DIR_CANDIDATES[split]
        subdirs = {d.name.lower(): d for d in self.root.iterdir() if d.is_dir()}
        for candidate in candidates:
            if candidate in subdirs:
                return subdirs[candidate]
        raise FileNotFoundError(
            f"В каталоге {self.root} не найдена папка выборки '{split}' (искали: {', '.join(candidates)}). "
            f"Ожидаемая структура: {self.root}/trainval/... и {self.root}/test/..."
        )

    def _find_class_dirs(self, split_dir: Path) -> List[Path]:
        """Найти все каталоги-классы в выборке (рекурсивный обход).

        Классом считается каталог, непосредственно содержащий хотя бы один
        файл изображения; промежуточные каталоги-поднаборы (без изображений)
        классами не считаются. Скрытые каталоги (имя начинается с точки,
        например .ipynb_checkpoints) пропускаются. Порядок результата
        детерминирован: сортировка по относительному пути.
        """
        extensions = tuple(self.IMAGE_EXTENSIONS)
        class_dirs: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(split_dir):
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
            if any(name.lower().endswith(extensions) for name in filenames):
                class_dirs.append(Path(dirpath))
        if not class_dirs:
            raise FileNotFoundError(
                f"В {split_dir} не найдено ни одного каталога с изображениями поддерживаемых "
                f"форматов: {sorted(self.IMAGE_EXTENSIONS)}. Ожидаемая структура: "
                f"{split_dir}/<имя_класса>/*.jpg или {split_dir}/<поднабор>/<имя_класса>/*.jpg"
            )
        class_dirs.sort(key=lambda p: p.relative_to(split_dir).as_posix().lower())
        return class_dirs

    def _discover_class_names(self, split_dir: Path) -> List[str]:
        """Список имён классов выборки (алфавитно отсортированный, без повторов)."""
        return sorted({d.name for d in self._find_class_dirs(split_dir)})

    # ------------------------------------------------------------------
    # Загрузка выборки
    # ------------------------------------------------------------------

    def load(self, split: str) -> List[Sample]:
        """Загрузить выборку: вернуть список пар (путь к изображению, метка класса).

        Классы тестовой выборки проверяются на совпадение с trainval:
        при несовпадении выбрасывается ошибка с перечнем отсутствующих
        и лишних классов (тихое смешивание меток недопустимо).
        """
        self._check_split(split)
        split_dir = self.split_dirs[split]
        class_dirs = self._find_class_dirs(split_dir)
        found_names = sorted({d.name for d in class_dirs})
        if set(found_names) != set(self._classes):
            missing = sorted(set(self._classes) - set(found_names))
            extra = sorted(set(found_names) - set(self._classes))
            raise ValueError(
                f"Классы в '{split_dir}' не совпадают с классами trainval. "
                f"Отсутствуют: {missing or 'нет'}; лишние: {extra or 'нет'}. "
                f"Набор классов (имена каталогов с изображениями) должен быть одинаковым "
                f"в trainval и test."
            )
        # файлы класса собираются со всех поднаборов, порядок детерминирован
        files_by_class: Dict[str, List[Path]] = {name: [] for name in self._classes}
        for class_dir in class_dirs:
            files = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
            )
            files_by_class[class_dir.name].extend(files)
        samples: List[Sample] = []
        for label, class_name in enumerate(self._classes):
            class_files = sorted(files_by_class[class_name])
            if self.max_per_class > 0:
                class_files = class_files[: self.max_per_class]
            samples.extend((path, label) for path in class_files)
            logger.info("  класс '%s': %d изображений", class_name, len(class_files))
        if not samples:
            raise ValueError(
                f"В {split_dir} не найдено изображений поддерживаемых форматов: {sorted(self.IMAGE_EXTENSIONS)}"
            )
        logger.info("Выборка '%s': %d изображений, %d классов", split, len(samples), len(self._classes))
        return samples

    @property
    def class_names(self) -> List[str]:
        return list(self._classes)


# Псевдоним для совместимости с прежней версией кода и документации
ArchitectureDataLoader = FolderDataLoader
