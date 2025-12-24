"""
Реализация алгоритма "Мешок слов" (Bag of Visual Words)
Использует SIFT/ORB/AKAZE детекторы и SVM классификатор
"""

import os
import cv2
import numpy as np
import joblib
import json
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from base_classifier import BaseClassifier

class BOWClassifier(BaseClassifier):
    """Классификатор на основе алгоритма 'Мешок слов'."""
    
    def __init__(self, n_clusters=200, image_size=(224, 224), detector_type='sift'):
        super().__init__('bow', image_size)
        self.n_clusters = n_clusters
        self.detector_type = detector_type
        self.kmeans = None
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Инициализация детектора
        if detector_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif detector_type == 'akaze':
            self.detector = cv2.AKAZE_create()
        else:
            self.detector = cv2.SIFT_create()
    
    def extract_descriptors(self, image_path):
        """Извлекает дескрипторы из изображения."""
        image = cv2.imread(image_path)
        if image is None:
            return np.array([])
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return descriptors if descriptors is not None else np.array([])
    
    def build_vocabulary(self, all_descriptors):
        """Строит словарь визуальных слов."""
        all_descriptors = np.vstack(all_descriptors)
        
        # Используем выборку для ускорения
        if len(all_descriptors) > 10000:
            indices = np.random.choice(len(all_descriptors), 10000, replace=False)
            all_descriptors = all_descriptors[indices]
        
        print(f"Построение словаря из {len(all_descriptors)} дескрипторов...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(all_descriptors)
        print(f"Словарь из {self.n_clusters} визуальных слов построен")
        return self.kmeans
    
    def image_to_histogram(self, descriptors):
        """Преобразует дескрипторы в гистограмму визуальных слов."""
        if len(descriptors) == 0:
            return np.zeros(self.n_clusters)
        
        labels = self.kmeans.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=self.n_clusters, range=(0, self.n_clusters))
        histogram = histogram.astype(np.float32)
        
        if np.sum(histogram) > 0:
            histogram /= np.sum(histogram)
        
        return histogram
    
    def train(self, train_file, images_dir):
        """Обучение модели BOW."""
        print("=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ 'МЕШОК СЛОВ'")
        print(f"Детектор: {self.detector_type}, Кластеров: {self.n_clusters}")
        print("=" * 60)
        
        # Загрузка данных
        train_paths, train_labels, train_ids = self.load_data(train_file, images_dir)
        
        # Извлечение дескрипторов
        print("\n1. Извлечение локальных признаков...")
        all_descriptors = []
        valid_indices = []
        
        for i, image_path in enumerate(train_paths):
            descriptors = self.extract_descriptors(image_path)
            if len(descriptors) > 0:
                all_descriptors.append(descriptors)
                valid_indices.append(i)
        
        print(f"Извлечено {sum(len(d) for d in all_descriptors)} дескрипторов")
        
        # Построение словаря
        print("\n2. Построение словаря визуальных слов...")
        self.build_vocabulary(all_descriptors)
        
        # Преобразование в гистограммы
        print("\n3. Преобразование изображений в гистограммы...")
        train_features = []
        train_labels_filtered = []
        train_ids_filtered = []
        
        for idx in valid_indices:
            descriptors = self.extract_descriptors(train_paths[idx])
            histogram = self.image_to_histogram(descriptors)
            train_features.append(histogram)
            train_labels_filtered.append(train_labels[idx])
            train_ids_filtered.append(train_ids[idx])
        
        train_features = np.array(train_features)
        
        # Масштабирование признаков
        print("\n4. Масштабирование признаков...")
        train_features_scaled = self.scaler.fit_transform(train_features)
        
        # Обучение классификатора
        print("\n5. Обучение классификатора...")
        self.classifier = SVC(kernel='linear', probability=True, random_state=42, C=1.0)
        self.classifier.fit(train_features_scaled, train_ids_filtered)
        
        # Оценка на обучающей выборке
        print("\n6. Оценка качества...")
        train_predictions = self.classifier.predict(train_features_scaled)
        train_predictions_labels = [self.id_to_label[pred] for pred in train_predictions]
        
        accuracy = self.evaluate(train_labels_filtered, train_predictions_labels, "обучающей выборке")

        self.plot_confusion_matrix(train_ids_filtered, train_predictions, 
                                  "Матрица ошибок BOW (обучающая выборка)")

        return accuracy
    
    def test(self, test_file, images_dir):
        """Тестирование модели BOW."""
        print("=" * 60)
        print("ТЕСТИРОВАНИЕ МОДЕЛИ 'МЕШОК СЛОВ'")
        print("=" * 60)
        
        # Загрузка данных
        test_paths, test_labels, test_ids = self.load_data(test_file, images_dir)
        
        # Преобразование тестовых изображений
        test_features = []
        valid_indices = []
        
        for i, path in enumerate(test_paths):
            descriptors = self.extract_descriptors(path)
            if len(descriptors) > 0:
                histogram = self.image_to_histogram(descriptors)
                test_features.append(histogram)
                valid_indices.append(i)
        
        test_features = np.array(test_features)
        
        # Масштабирование и предсказание
        test_features_scaled = self.scaler.transform(test_features)
        test_predictions = self.classifier.predict(test_features_scaled)
        
        # Фильтрация меток
        test_labels_filtered = [test_labels[i] for i in valid_indices]
        test_ids_filtered = [test_ids[i] for i in valid_indices]
        test_predictions_labels = [self.id_to_label[pred] for pred in test_predictions]
        
        # Оценка качества
        accuracy = self.evaluate(test_labels_filtered, test_predictions_labels, "тестовой выборке")
        
        # Матрица ошибок
        self.plot_confusion_matrix(test_ids_filtered, test_predictions, 
                                  "Матрица ошибок BOW (тестовая выборка)")
        
        return accuracy
    
    def predict_single(self, image_path):
        """Предсказание класса для одного изображения."""
        descriptors = self.extract_descriptors(image_path)
        histogram = self.image_to_histogram(descriptors)
        histogram_scaled = self.scaler.transform([histogram])
        
        prediction_id = self.classifier.predict(histogram_scaled)[0]
        probabilities = self.classifier.predict_proba(histogram_scaled)[0]
        
        prediction_label = self.id_to_label[prediction_id]
        confidence = probabilities[prediction_id]
        
        return prediction_label, confidence
    
    def visualize_features(self, image_path, output_path=None):
        """Визуализирует ключевые точки на изображении."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = self.detector.detect(gray, None)
        
        # Настраиваем флаги рисования в зависимости от детектора
        if self.detector_type == 'sift':
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        else:
            flags = 0
        
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, 
            flags=flags,
            color=(0, 255, 0)
        )
        
        print(f"Найдено {len(keypoints)} ключевых точек")
        
        # Всегда показываем изображение
        import matplotlib.pyplot as plt
        
        # Создаем фигуру с двумя субплогами
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Оригинальное изображение
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Оригинальное изображение')
        axes[0].axis('off')
        
        # Изображение с ключевыми точками
        result_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        axes[1].imshow(result_rgb)
        axes[1].set_title(f'Ключевые точки ({self.detector_type.upper()})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return image_with_keypoints
    
    def save_model(self, model_dir):
        """Сохраняет модель BOW."""
        model_dir = os.path.join(model_dir, 'bow_model')
        super().save_model(model_dir)
        
        joblib.dump(self.kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
        joblib.dump(self.classifier, os.path.join(model_dir, 'classifier_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        detector_params = {
            'detector_type': self.detector_type,
            'n_clusters': self.n_clusters
        }
        
        with open(os.path.join(model_dir, 'detector_params.json'), 'w') as f:
            json.dump(detector_params, f, indent=2)
    
    def load_model(self, model_dir):
        """Загружает модель BOW."""
        model_dir = os.path.join(model_dir, 'bow_model')
        super().load_model(model_dir)
        
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
        self.classifier = joblib.load(os.path.join(model_dir, 'classifier_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'detector_params.json'), 'r') as f:
            detector_params = json.load(f)
        
        self.detector_type = detector_params['detector_type']
        self.n_clusters = detector_params['n_clusters']
        
        if self.detector_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif self.detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'akaze':
            self.detector = cv2.AKAZE_create()
        
        return True
