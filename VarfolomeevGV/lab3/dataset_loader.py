"""
Модуль для загрузки и предобработки данных для классификации изображений.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path


class DatasetLoader:
    """Класс для загрузки и предобработки набора данных изображений."""
    
    def __init__(self, data_dir: str = "NNClassification", img_size: Tuple[int, int] = (224, 224)):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_dir: Путь к директории с данными
            img_size: Размер изображений после предобработки (ширина, высота)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # Маппинг классов
        self.class_mapping = {
            "01_NizhnyNovgorodKremlin": 0,
            "04_ArkhangelskCathedral": 1,
            "08_PalaceOfLabor": 2
        }
        
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.num_classes = len(self.class_mapping)
    
    def load_dataset(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Загрузка всех изображений из директорий ExtDataset и NNSUDataset.
        
        Returns:
            Словарь с ключами 'ExtDataset' и 'NNSUDataset', значения - списки кортежей (путь, класс)
        """
        dataset = {"ExtDataset": [], "NNSUDataset": []}
        
        for dataset_name in ["ExtDataset", "NNSUDataset"]:
            dataset_path = self.data_dir / dataset_name
            
            if not dataset_path.exists():
                print(f"Предупреждение: директория {dataset_path} не найдена")
                continue
            
            # Проходим по всем классам
            for class_name, class_id in self.class_mapping.items():
                class_path = dataset_path / class_name
                
                if not class_path.exists():
                    continue
                
                # Загружаем все изображения из директории класса
                for img_file in class_path.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dataset[dataset_name].append((str(img_file), class_id))
        
        return dataset
    
    def split_train_test(self, train_file: str = "train.txt") -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Разбиение данных на тренировочную и тестовую выборки на основе train.txt.
        
        Args:
            train_file: Путь к файлу с перечнем тренировочных изображений
            
        Returns:
            Кортеж (train_data, test_data), где каждый элемент - список кортежей (путь, класс)
        """
        # Загружаем все данные
        all_data = self.load_dataset()
        
        # Читаем список тренировочных файлов
        train_paths = set()
        train_file_path = Path(train_file)
        
        if train_file_path.exists():
            with open(train_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    path = line.strip()
                    if path:
                        # Нормализуем путь (заменяем обратные слеши на прямые для кроссплатформенности)
                        normalized_path = path.replace('\\', os.sep)
                        # Добавляем полный путь и нормализуем его
                        full_path = (self.data_dir / normalized_path).resolve()
                        train_paths.add(str(full_path))
        else:
            print(f"Предупреждение: файл {train_file} не найден, все данные будут использованы для теста")
        
        # Разделяем на train и test
        train_data = []
        test_data = []
        
        for dataset_name, images in all_data.items():
            for img_path, class_id in images:
                # Нормализуем путь для корректного сравнения
                normalized_img_path = str(Path(img_path).resolve())
                if normalized_img_path in train_paths:
                    train_data.append((img_path, class_id))
                else:
                    test_data.append((img_path, class_id))
        
        print(f"Загружено тренировочных изображений: {len(train_data)}")
        print(f"Загружено тестовых изображений: {len(test_data)}")
        
        return train_data, test_data
    
    def preprocess_image(self, img_path: str, normalize: bool = True) -> np.ndarray:
        """
        Предобработка изображения.
        
        Args:
            img_path: Путь к изображению
            normalize: Нужно ли нормализовать значения пикселей [0, 1]
            
        Returns:
            Предобработанное изображение
        """
        # Загружаем изображение
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        # Конвертируем BGR в RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Изменяем размер
        img = cv2.resize(img, self.img_size)
        
        # Нормализация
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_images(self, data_list: List[Tuple[str, int]], normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загрузка и предобработка списка изображений.
        
        Args:
            data_list: Список кортежей (путь, класс)
            normalize: Нужно ли нормализовать значения пикселей
            
        Returns:
            Кортеж (images, labels), где images - массив изображений, labels - массив меток
        """
        images = []
        labels = []
        
        for img_path, class_id in data_list:
            try:
                img = self.preprocess_image(img_path, normalize=normalize)
                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"Ошибка при загрузке {img_path}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    def get_class_name(self, class_id: int) -> str:
        """Получить имя класса по его ID."""
        return self.reverse_class_mapping.get(class_id, "Unknown")

