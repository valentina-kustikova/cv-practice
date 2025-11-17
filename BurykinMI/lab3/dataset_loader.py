import os
from typing import List, Tuple


# ============================================================================
# Загрузчик датасета
# ============================================================================

class DatasetLoader:
    """Класс для загрузки и управления датасетом"""

    FOLDER_TO_CLASS = {
        '01_NizhnyNovgorodKremlin': 'kremlin',
        '04_ArkhangelskCathedral': 'cathedral',
        '08_PalaceOfLabor': 'palace'
    }

    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    def __init__(self, data_dir: str, split_file: str):
        """
        Args:
            data_dir: Путь к директории с данными
            split_file: Файл с разбиением на train/test
        """
        self.data_dir = data_dir
        self.split_file = split_file

    def load(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Загрузить датасет

        Returns:
            train_data, train_labels, test_data, test_labels
        """
        train_files_set = self._read_split_file()
        all_images = self._collect_all_images()

        train_data, train_labels, test_data, test_labels = self._split_dataset(
            all_images, train_files_set
        )

        self._print_statistics(train_labels, test_labels)

        return train_data, train_labels, test_data, test_labels

    def _read_split_file(self) -> set:
        """Прочитать файл разбиения"""
        with open(self.split_file, 'r', encoding='utf-8') as f:
            train_files = [line.strip() for line in f if line.strip()]
        return set(train_files)

    def _collect_all_images(self) -> List[Tuple[str, str, str]]:
        """Собрать все изображения из датасета"""
        all_images = []
        dataset_folders = ['ExtDataset', 'NNSUDataset']

        for dataset_folder in dataset_folders:
            dataset_path = os.path.join(self.data_dir, dataset_folder)

            if not os.path.exists(dataset_path):
                print(f"Предупреждение: директория {dataset_path} не найдена")
                continue

            for class_folder in os.listdir(dataset_path):
                class_folder_path = os.path.join(dataset_path, class_folder)

                if not os.path.isdir(class_folder_path):
                    continue

                if class_folder not in self.FOLDER_TO_CLASS:
                    print(f"Предупреждение: неизвестная папка {class_folder}, пропускаем")
                    continue

                class_name = self.FOLDER_TO_CLASS[class_folder]

                for img_file in os.listdir(class_folder_path):
                    if img_file.lower().endswith(self.VALID_EXTENSIONS):
                        full_path = os.path.join(class_folder_path, img_file)
                        rel_path = os.path.join(dataset_folder, class_folder, img_file)
                        all_images.append((full_path, class_name, rel_path))

        return all_images

    def _split_dataset(
            self,
            all_images: List[Tuple[str, str, str]],
            train_files_set: set
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Разделить датасет на train и test"""
        train_data, train_labels = [], []
        test_data, test_labels = [], []

        for full_path, class_name, rel_path in all_images:
            if rel_path in train_files_set:
                train_data.append(full_path)
                train_labels.append(class_name)
            else:
                test_data.append(full_path)
                test_labels.append(class_name)

        return train_data, train_labels, test_data, test_labels

    def _print_statistics(self, train_labels: List[str], test_labels: List[str]) -> None:
        """Вывести статистику датасета"""
        class_names = sorted(list(set(train_labels + test_labels)))

        print(f"Загружено {len(train_labels)} тренировочных и {len(test_labels)} тестовых изображений")
        print(f"Классы: {class_names}")
        print(f"Распределение тренировочных данных:")
        for class_name in class_names:
            count = train_labels.count(class_name)
            print(f"  {class_name}: {count}")
        print(f"Распределение тестовых данных:")
        for class_name in class_names:
            count = test_labels.count(class_name)
            print(f"  {class_name}: {count}")
