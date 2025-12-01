import os
from typing import List

import cv2
import joblib
import numpy as np
# --- ИЗМЕНЕНИЕ: Импортируем MiniBatchKMeans ---
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BagOfWordsClassifier:
    """
    Реализация классификатора на основе алгоритма "Мешок Слов".
    """

    def __init__(self, n_clusters: int = 100, descriptor_type: str = 'SIFT'):
        self.n_clusters = n_clusters
        self.descriptor_type = descriptor_type
        self.model_path = 'models/bow_params.pkl'

        self.kmeans = None
        self.classifier = None
        self.scaler = None

        if descriptor_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif descriptor_type == 'ORB':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError("Неподдерживаемый дескриптор. Используйте 'SIFT' или 'ORB'.")

    def _extract_features(self, image_path: str) -> np.ndarray:
        """Извлекает дескрипторы из одного изображения."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        _, des = self.detector.detectAndCompute(img, None)
        return des

    def _get_global_descriptors(self, image_paths: List[str], is_training: bool) -> List[np.ndarray]:
        """Формирует гистограммы визуальных слов."""
        all_descriptors = []
        descriptors_per_image = []

        # 1. Извлечение дескрипторов
        for path in image_paths:
            des = self._extract_features(path)
            if des is not None and len(des) > 0:
                all_descriptors.append(des)
                descriptors_per_image.append(des)

        # Шаг 2: Кластеризация для создания словаря (только при обучении)
        if is_training:
            all_des_stacked = np.concatenate(all_descriptors)
            print(f"Начало кластеризации {all_des_stacked.shape[0]} дескрипторов...")

            # Используем MiniBatchKMeans
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                n_init='auto',  # Автоматическое определение инициализации
                batch_size=10000,  # Размер батча для ускорения
                random_state=42,
                verbose=1  # Включаем вывод прогресса
            )
            self.kmeans.fit(all_des_stacked)
            print("Кластеризация завершена. Словарь визуальных слов создан.")

        # Шаг 3: Создание гистограмм признаков
        histograms = []
        for des in descriptors_per_image:
            # Предсказание, к какому визуальному слову относится каждый дескриптор
            words = self.kmeans.predict(des)
            # Создание гистограммы частот визуальных слов (Bag of Words)
            histogram, _ = np.histogram(words, bins=range(self.n_clusters + 1), density=True)
            histograms.append(histogram)

        return np.array(histograms)

    def train(self, train_paths: List[str], train_labels: List[int], visualize: bool = False):
        """Обучает классификатор. Включает создание словаря, извлечение гистограмм и обучение SVM."""

        print("--- Обучение Bag-of-Words (BoW) классификатора ---")

        # 1. Извлечение дескрипторов и создание словаря визуальных слов
        train_features = self._get_global_descriptors(train_paths, is_training=True)

        # 2. Масштабирование признаков (важно для SVM)
        self.scaler = StandardScaler().fit(train_features)
        train_features_scaled = self.scaler.transform(train_features)

        # 3. Обучение классификатора (Support Vector Machine)
        print("Обучение SVM классификатора...")
        self.classifier = SVC(kernel='linear', C=1.0, random_state=42)
        self.classifier.fit(train_features_scaled, train_labels)
        print("Обучение SVM завершено.")

        # 4. Сохранение
        self._save_params()

        # 5. Визуализация первых пяти картинок
        if visualize:
            num_visualize = min(5, len(train_paths))
            print(f"\nНачало визуализации ключевых точек для {num_visualize} изображений...")
            for i in range(num_visualize):
                self._visualize_features(train_paths[i], index=i + 1)

    def _save_params(self):
        """Сохраняет обученные параметры BoW."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        params = {
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'scaler': self.scaler
        }
        joblib.dump(params, self.model_path)
        print(f"Параметры BoW сохранены в: {self.model_path}")

    def _load_params(self):
        """Загружает обученные параметры BoW."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл параметров BoW не найден: {self.model_path}. Запустите в режиме 'train'.")

        params = joblib.load(self.model_path)
        self.kmeans = params['kmeans']
        self.classifier = params['classifier']
        self.scaler = params['scaler']
        print(f"Параметры BoW успешно загружены из: {self.model_path}")

    def predict(self, test_paths: List[str]) -> List[int]:
        """Предсказывает метки классов для тестовых изображений."""

        if self.kmeans is None:
            self._load_params()

            # 1. Извлечение гистограмм (используя обученный self.kmeans)
        test_features = self._get_global_descriptors(test_paths, is_training=False)

        # 2. Масштабирование
        test_features_scaled = self.scaler.transform(test_features)

        # 3. Предсказание
        predictions = self.classifier.predict(test_features_scaled)
        return predictions.tolist()

    def _visualize_features(self, image_path: str, index: int = 1):
        """Визуализация ключевых точек."""
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, _ = self.detector.detectAndCompute(gray_img, None)
        img_kp = cv2.drawKeypoints(gray_img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Создаем директорию 'visualized', если она не существует
        vis_dir = 'visualized'
        os.makedirs(vis_dir, exist_ok=True)
        # Сохраняем в новую директорию с уникальным именем
        base_name = os.path.basename(image_path)
        vis_path = os.path.join(vis_dir, f'keypoints_viz_{index}_{base_name}')

        cv2.imwrite(vis_path, img_kp)
        print(f"Визуализация #{index} сохранена в {vis_path}")
