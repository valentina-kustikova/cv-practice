import os
import pickle
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC

from abstract import FeatureExtractor, ClassificationStrategy


class BagOfWordsStrategy(ClassificationStrategy):
    """Стратегия классификации на основе мешка визуальных слов и SVM"""

    def __init__(self, feature_extractor: FeatureExtractor, n_clusters: int = 300):
        self.feature_extractor = feature_extractor
        # n_clusters - размер словаря. Описываем картинки набором из 300 "типичных" признаков.
        self.n_clusters = n_clusters
        self.kmeans = None
        self.classifier = None
        self.class_names = None

    def _build_vocabulary(self, image_paths: List[str]) -> None:
        """Построить словарь визуальных слов (обучение K-Means)"""
        print("Извлечение дескрипторов...")
        all_descriptors = []

        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(image_paths)} изображений")
            # Извлекаем дескрипторы (SIFT).
            # descriptors shape: [N, 128], где N - кол-во найденных точек (разное для каждой картинки)
            descriptors = self.feature_extractor.extract(img_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        # np.vstack: Объединяем дескрипторы всех картинок в одну общую матрицу.
        # Shape: [Total_Points, 128]. Мы теряем связь с конкретными картинками,
        # чтобы найти общие паттерны во всем датасете.
        all_descriptors = np.vstack(all_descriptors)
        print(f"Всего дескрипторов: {len(all_descriptors)}")

        print(f"Построение словаря из {self.n_clusters} слов...")
        # Кластеризация: находим 300 центров сгустков в 128-мерном пространстве.
        # Эти центры (centroids) становятся нашими "Визуальными словами".
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1000)
        self.kmeans.fit(all_descriptors)
        print("Словарь построен!")

    def _get_bow_representation(self, image_path: str) -> np.ndarray:
        """Получить BoW-представление (Векторизация изображения)"""
        descriptors = self.feature_extractor.extract(image_path) # [N, 128]

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        # КВАНТОВАНИЕ:
        # Каждому из N дескрипторов сопоставляем индекс ближайшего кластера (0..299).
        # Мы заменяем сложные векторы [128] на простые числа (ID слова).
        labels = self.kmeans.predict(descriptors)

        # Построение гистограммы:
        # Считаем, сколько раз встретилось каждое "слово".
        # Результат: вектор фиксированной длины [300,].
        histogram = np.zeros(self.n_clusters)
        for label in labels:
            histogram[label] += 1

        # НОРМАЛИЗАЦИЯ (L1-norm):
        # Делим на общее число точек. Получаем частоты (вероятности).
        # Это делает результат независимым от разрешения картинки и кол-ва найденных точек.
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)

        return histogram

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Обучить модель BoW + SVM"""
        self.class_names = sorted(list(set(train_labels)))

        # 1. Строим словарь признаков (обучаем K-Means)
        self._build_vocabulary(train_data)

        # 2. Преобразуем картинки в векторы-гистограммы
        print("Извлечение BoW-признаков...")
        X_train = []
        for img_path in train_data:
            # Превращаем картинку в вектор [300,]
            bow_features = self._get_bow_representation(img_path)
            X_train.append(bow_features)

        # Итоговая матрица обучения X_train shape: [M_samples, 300]
        X_train = np.array(X_train)

        # 3. Обучаем SVM разделять эти 300-мерные векторы по классам.
        # kernel='rbf' позволяет строить нелинейные границы.
        print("Обучение SVM...")
        self.classifier = SVC(kernel='rbf', C=10, gamma='scale')
        self.classifier.fit(X_train, train_labels)
        print("Обучение завершено!")

    def predict(self, image_path: str) -> str:
        """Предсказать класс"""
        # Превращаем новую картинку в вектор [300,] используя тот же словарь
        bow_features = self._get_bow_representation(image_path)
        # SVM определяет, к какому классу относится точка в 300-мерном пространстве
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
