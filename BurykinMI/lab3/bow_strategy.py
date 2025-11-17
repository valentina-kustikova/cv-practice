import os
import pickle
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC

from abstract import FeatureExtractor, ClassificationStrategy


# ============================================================================
# Стратегия Bag of Words + SVM
# ============================================================================

class BagOfWordsStrategy(ClassificationStrategy):
    """Стратегия классификации на основе мешка визуальных слов и SVM"""

    def __init__(self, feature_extractor: FeatureExtractor, n_clusters: int = 300):
        """
        Args:
            feature_extractor: Экстрактор признаков
            n_clusters: Количество кластеров для словаря
        """
        self.feature_extractor = feature_extractor
        self.n_clusters = n_clusters
        self.kmeans = None
        self.classifier = None
        self.class_names = None

    def _build_vocabulary(self, image_paths: List[str]) -> None:
        """Построить словарь визуальных слов"""
        print("Извлечение дескрипторов из тренировочных изображений...")
        all_descriptors = []

        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(image_paths)} изображений")

            descriptors = self.feature_extractor.extract(img_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        all_descriptors = np.vstack(all_descriptors)
        print(f"Всего дескрипторов: {len(all_descriptors)}")

        print(f"Построение словаря из {self.n_clusters} визуальных слов...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            verbose=1
        )
        self.kmeans.fit(all_descriptors)
        print("Словарь построен!")

    def _get_bow_representation(self, image_path: str) -> np.ndarray:
        """Получить BoW-представление изображения"""
        descriptors = self.feature_extractor.extract(image_path)

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        labels = self.kmeans.predict(descriptors)

        histogram = np.zeros(self.n_clusters)
        for label in labels:
            histogram[label] += 1

        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)

        return histogram

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Обучить модель BoW + SVM"""
        self.class_names = sorted(list(set(train_labels)))

        # Построение словаря
        self._build_vocabulary(train_data)

        # Извлечение BoW-признаков
        print("Извлечение BoW-признаков...")
        X_train = []
        for i, img_path in enumerate(train_data):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(train_data)} изображений")
            bow_features = self._get_bow_representation(img_path)
            X_train.append(bow_features)

        X_train = np.array(X_train)

        # Обучение SVM
        print("Обучение SVM классификатора...")
        self.classifier = SVC(kernel='rbf', C=10, gamma='scale', verbose=True)
        self.classifier.fit(X_train, train_labels)
        print("Обучение завершено!")

    def predict(self, image_path: str) -> str:
        """Предсказать класс изображения"""
        bow_features = self._get_bow_representation(image_path)
        prediction = self.classifier.predict([bow_features])[0]
        return prediction

    def save(self, filepath: str) -> None:
        """Сохранить модель"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'feature_extractor_name': self.feature_extractor.get_name(),
            'n_clusters': self.n_clusters,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'class_names': self.class_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {filepath}")

    def load(self, filepath: str) -> None:
        """Загрузить модель"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.n_clusters = model_data['n_clusters']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.class_names = model_data['class_names']

        print(f"Модель загружена из {filepath}")

    def get_params(self) -> Dict[str, Any]:
        """Получить параметры модели"""
        return {
            'algorithm': 'bow',
            'detector': self.feature_extractor.get_name(),
            'n_clusters': self.n_clusters
        }
