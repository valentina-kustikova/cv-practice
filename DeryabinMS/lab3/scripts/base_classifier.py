"""
Базовый класс классификатора изображений
Реализует общий функционал для всех алгоритмов классификации
"""

import os
import json
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class BaseClassifier(ABC):
    """Абстрактный базовый класс для классификаторов изображений."""
    
    def __init__(self, algorithm, image_size=(224, 224)):
        self.algorithm = algorithm
        self.image_size = image_size
        self.class_names = ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль']
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_label = {i: name for name, i in self.label_to_id.items()}
        self.model = None
    
    def detect_label_from_path(self, file_path):
        """Определяет метку класса из пути к файлу."""
        file_path_lower = file_path.lower()
        
        if '01_nizhnynovgorodkremlin' in file_path_lower or 'нижегородский кремль' in file_path_lower:
            return 'Нижегородский Кремль'
        elif '08_palaceoflabor' in file_path_lower or 'дворец труда' in file_path_lower:
            return 'Дворец труда'
        elif '04_arkhangelskcathedral' in file_path_lower or 'архангельский собор' in file_path_lower:
            return 'Архангельский собор'
        return None
    
    def load_data(self, file_list_path, images_dir):
        """Загружает данные из файла со списком изображений."""
        with open(file_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        image_paths, labels, label_ids = [], [], []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            full_path = os.path.join(images_dir, line)
            label = self.detect_label_from_path(full_path)
            
            if label is not None and os.path.exists(full_path):
                image_paths.append(full_path)
                labels.append(label)
                label_ids.append(self.label_to_id[label])
        
        print(f"Загружено {len(image_paths)} изображений")
        for label_name in self.class_names:
            count = labels.count(label_name)
            print(f"  {label_name}: {count} изображений")
        
        return image_paths, labels, label_ids
    
    def evaluate(self, true_labels, pred_labels, dataset_type="выборке"):
        """Оценка качества классификации."""
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Accuracy на {dataset_type}: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print("\nДетальный отчет:")
        print(classification_report(true_labels, pred_labels, target_names=self.class_names))
        return accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Матрица ошибок"):
        """Визуализация матрицы ошибок."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(len(self.class_names)),
               yticks=np.arange(len(self.class_names)),
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
    
    def save_model(self, model_dir):
        """Сохраняет модель в указанную директорию."""
        os.makedirs(model_dir, exist_ok=True)
        
        metadata = {
            'algorithm': self.algorithm,
            'image_size': self.image_size,
            'class_names': self.class_names,
            'label_to_id': self.label_to_id
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Модель сохранена в {model_dir}")
    
    def load_model(self, model_dir):
        """Загружает модель из указанной директории."""
        with open(os.path.join(model_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.algorithm = metadata['algorithm']
        self.class_names = metadata['class_names']
        self.image_size = tuple(metadata['image_size'])
        self.label_to_id = metadata['label_to_id']
        self.id_to_label = {int(i): name for name, i in self.label_to_id.items()}
        
        print(f"Модель загружена из {model_dir}")
        return True
    
    @abstractmethod
    def train(self, train_file, images_dir):
        """Абстрактный метод для обучения модели."""
        pass
    
    @abstractmethod
    def test(self, test_file, images_dir):
        """Абстрактный метод для тестирования модели."""
        pass
    
    @abstractmethod
    def predict_single(self, image_path):
        """Абстрактный метод для предсказания класса одного изображения."""
        pass
