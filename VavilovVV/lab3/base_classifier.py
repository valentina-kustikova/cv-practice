# base_classifier.py

import logging
import argparse
from pathlib import Path
from typing import List, Tuple
from abc import ABC, abstractmethod

log = logging.getLogger()


class BaseClassifier(ABC):
    """
    Абстрактный базовый класс для классификаторов изображений.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.train_file = Path(args.train_file)
        self.test_file = Path(args.test_file)

        self.classes = args.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.model_save_dir = Path(args.model_dir)
        self.model_save_dir.mkdir(exist_ok=True)

    def load_data(self, file_path: Path) -> Tuple[List[str], List[int]]:
        """
        Загружает пути к изображениям и их метки из указанного файла.
        """
        image_paths = []
        labels = []

        if not file_path.exists():
            log.error(f"Файл не найден: {file_path}")
            return image_paths, labels

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip().replace('\\', '/')
                if not line:
                    continue

                parts = line.split('/')
                if len(parts) < 2:
                    log.warning(f"Строка {line_num}: Неверный путь: {line}")
                    continue

                # parts[1] - имя класса
                class_name = parts[1]
                img_path = self.data_dir / line

                if not img_path.exists():
                    log.warning(f"Строка {line_num}: Файл не найден: {img_path}")
                    continue

                if class_name not in self.class_to_idx:
                    log.warning(f"Строка {line_num}: Неизвестный класс '{class_name}'.")
                    continue

                label = self.class_to_idx[class_name]
                image_paths.append(str(img_path))
                labels.append(label)

        log.info(f"Загружено {len(image_paths)} ссылок на изображения из {file_path}")
        return image_paths, labels

    @abstractmethod
    def train(self):
        """
        Абстрактный метод для обучения модели.
        """
        pass

    @abstractmethod
    def test(self):
        """
        Абстрактный метод для тестирования модели.
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Абстрактный метод для сохранения модели.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Абстрактный метод для загрузки модели.
        """
        pass