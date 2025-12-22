import os
import json
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class BaseClassifier(ABC):
    """Базовый абстрактный класс для всех классификаторов изображений."""
    
    def __init__(self, algorithm, image_size=(224, 224), class_names=None):
        self.algorithm = algorithm
        self.image_size = image_size
        
        if class_names is None:
            self.class_names = ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль']
        else:
            self.class_names = class_names
            
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_label = {i: name for name, i in self.label_to_id.items()}
        
    def detect_label_from_path(self, file_path):
        """Определяет метку класса из пути к файлу."""
        file_path_lower = file_path.lower()
        
        if '01_nizhnynovgorodkremlin' in file_path_lower or 'нижегородский кремль' in file_path_lower:
            return 'Нижегородский Кремль'
        elif '08_palaceoflabor' in file_path_lower or 'дворец труда' in file_path_lower:
            return 'Дворец труда'
        elif '04_arkhangelskcathedral' in file_path_lower or 'архангельский собор' in file_path_lower:
            return 'Архангельский собор'
        else:
            print(f"Предупреждение: не удалось определить метку для файла: {file_path}")
            return None

    def load_data(self, file_list_path, images_dir="."):
        """
        Загружает данные из файла со списком изображений.
        
        Args:
            file_list_path: путь к файлу со списком изображений (train.txt или test.txt)
            images_dir: корневая директория с изображениями
            
        Returns:
            image_paths: список путей к изображениям
            labels: список текстовых меток
            label_ids: список числовых идентификаторов меток
        """
        try:
            with open(file_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Ошибка: файл {file_list_path} не найден!")
            return [], [], []
        
        image_paths = []
        labels = []
        label_ids = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            full_path = os.path.join(images_dir, line)
            
            if os.path.exists(full_path):
                label = self.detect_label_from_path(full_path)
                if label is not None:
                    image_paths.append(full_path)
                    labels.append(label)
                    label_ids.append(self.label_to_id[label])
                else:
                    print(f"Пропущен файл (не определена метка): {full_path}")
            else:
                print(f"Предупреждение: файл не найден: {full_path}")
        
        print(f"Загружено {len(image_paths)} изображений из {file_list_path}")
        
        if len(image_paths) > 0:
            print("Распределение по классам:")
            for label_name in self.class_names:
                count = labels.count(label_name)
                if count > 0:
                    print(f"  {label_name}: {count} изображений")
        
        return image_paths, labels, label_ids

    def save_model(self, model_dir="models"):
        """Сохраняет метаданные модели в указанную директорию."""
        os.makedirs(model_dir, exist_ok=True)
        
        metadata = {
            'algorithm': self.algorithm,
            'image_size': self.image_size,
            'class_names': self.class_names,
            'label_to_id': self.label_to_id
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Метаданные модели сохранены в: {metadata_path}")

    def load_model(self, model_dir="models"):
        """Загружает метаданные модели из указанной директории."""
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Ошибка: файл метаданных не найден в {model_dir}")
            return False
        
        self.algorithm = metadata['algorithm']
        self.class_names = metadata['class_names']
        self.image_size = tuple(metadata['image_size'])
        self.label_to_id = metadata['label_to_id']
        self.id_to_label = {int(i): name for name, i in self.label_to_id.items()}
        
        print(f"Метаданные модели загружены из: {metadata_path}")
        print(f"Алгоритм: {self.algorithm}, Классы: {self.class_names}")
        return True

    def plot_confusion_matrix(self, y_true, y_pred, title="Матрица ошибок"):
        """Визуализирует матрицу ошибок классификации."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title=title,
               ylabel='Истинные метки',
               xlabel='Предсказанные метки')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.show()
        
        return cm

    @abstractmethod
    def train(self, train_file, images_dir=".", **kwargs):
        """Абстрактный метод для обучения модели."""
        pass

    @abstractmethod
    def test(self, test_file, images_dir=".", **kwargs):
        """Абстрактный метод для тестирования модели."""
        pass

    @abstractmethod
    def predict_single(self, image_path):
        """Абстрактный метод для предсказания класса одного изображения."""
        pass
