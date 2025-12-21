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
        self.n_clusters = n_clusters
        self.kmeans = None
        self.classifier = None
        self.class_names = None

    def _build_vocabulary(self, image_paths: List[str]) -> None:
        """Построить словарь визуальных слов (обучение K-Means)"""
        print("Извлечение дескрипторов из тренировочных изображений...")
        all_descriptors = []

        for i, img_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(image_paths)} изображений")

            # Получаем матрицу (N, 128) для SIFT, где N - кол-во точек на одной картинке
            descriptors = self.feature_extractor.extract(img_path)
            if descriptors is not None:
                all_descriptors.append(descriptors)

        # Сваливаем дескрипторы со всех картинок в одну кучу (вертикальный стек).
        # Мы теряем информацию о том, какой картинке принадлежала точка.
        # Получаем матрицу (Total_Points, 128).
        all_descriptors = np.vstack(all_descriptors)
        print(f"Всего дескрипторов: {len(all_descriptors)}")

        print(f"Построение словаря из {self.n_clusters} визуальных слов...")
        # Алгоритм ищет сгустки точек в 128-мерном пространстве признаков.
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            verbose=1
        )
        # Центры найденных кластеров становятся нашими "визуальными словами"
        self.kmeans.fit(all_descriptors)
        print("Словарь построен!")

    def _get_bow_representation(self, image_path: str) -> np.ndarray:
        """Получить BoW-представление изображения (Векторизация)"""
        descriptors = self.feature_extractor.extract(image_path)

        # Защита от черных квадратов или ошибок чтения
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        # КВАНТОВАНИЕ:
        # Для каждого дескриптора находим ближайшее "визуальное слово" (номер кластера).
        # Сложный вектор из 128 чисел заменяется одним числом (ID кластера).
        labels = self.kmeans.predict(descriptors)

        # Строим гистограмму частот.
        # Вектор длиной n_clusters, где каждое число - сколько раз встретилось "слово".
        histogram = np.zeros(self.n_clusters)
        for label in labels:
            histogram[label] += 1

        # НОРМАЛИЗАЦИЯ:
        # Делим на сумму, чтобы получить вероятности (частоты), а не абсолютные числа.
        if np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)

        return histogram

    def train(self, train_data: List[str], train_labels: List[str]) -> None:
        """Обучить модель BoW + SVM"""
        self.class_names = sorted(list(set(train_labels)))

        # 1. Формируем словарь признаков на основе всей выборки
        self._build_vocabulary(train_data)

        # 2. Переводим все картинки в векторы фиксированной длины (гистограммы)
        print("Извлечение BoW-признаков...")
        X_train = []
        for i, img_path in enumerate(train_data):
            if i % 10 == 0:
                print(f"Обработано {i}/{len(train_data)} изображений")
            bow_features = self._get_bow_representation(img_path)
            X_train.append(bow_features)

        X_train = np.array(X_train)

        # 3. Обучаем классификатор разделять векторы гистограмм.
        print("Обучение SVM классификатора...")
        self.classifier = SVC(kernel='rbf', C=10, gamma='scale', verbose=True)
        self.classifier.fit(X_train, train_labels)
        print("Обучение завершено!")

    def predict(self, image_path: str) -> str:
        """Предсказать класс изображения"""
        # Превращаем новую картинку в такой же вектор-гистограмму
        bow_features = self._get_bow_representation(image_path)
        # SVM определяет, по какую сторону гиперплоскости лежит этот вектор
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
