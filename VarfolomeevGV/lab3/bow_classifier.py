"""
Модуль реализации алгоритма мешок слов (BoW)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import pickle
import os


class BagOfWordsClassifier:
    """Класс для классификации изображений с использованием алгоритма "мешок слов"."""
    
    def __init__(self, vocab_size: int = 300, detector_type: str = "sift"):
        """
        Инициализация классификатора.
        
        Args:
            vocab_size: Размер словаря визуальных слов
            detector_type: Тип детектора/дескриптора ('sift', 'surf', 'orb')
        """
        self.vocab_size = vocab_size
        self.detector_type = detector_type.lower()
        self.detector = None
        self.vocabulary = None
        self.scaler = StandardScaler()
        self.classifier = None
        self.is_trained = False
        
        # Инициализация детектора
        self._init_detector()
    
    def _init_detector(self):
        """Инициализация детектора и дескриптора."""
        if self.detector_type == "sift":
            self.detector = cv2.SIFT_create()
        elif self.detector_type == "surf":
            # SURF требует дополнительной компиляции OpenCV с флагом OPENCV_ENABLE_NONFREE
            try:
                self.detector = cv2.xfeatures2d.SURF_create(400)
            except:
                print("SURF недоступен, используется SIFT")
                self.detector = cv2.SIFT_create()
                self.detector_type = "sift"
        elif self.detector_type == "orb":
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Неизвестный тип детектора: {self.detector_type}")
    
    def extract_features(self, img_path: str) -> np.ndarray:
        """
        Извлечение ключевых точек и дескрипторов из изображения.
        
        Args:
            img_path: Путь к изображению
            
        Returns:
            Массив дескрипторов
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        # Находим ключевые точки и дескрипторы
        if self.detector_type == "orb":
            keypoints, descriptors = self.detector.detectAndCompute(img, None)
        else:
            keypoints, descriptors = self.detector.detectAndCompute(img, None)
        
        # Если дескрипторы не найдены, возвращаем пустой массив
        if descriptors is None:
            return np.array([]).reshape(0, 128 if self.detector_type != "orb" else 32)
        
        return descriptors
    
    def build_vocabulary(self, train_data: List[Tuple[str, int]], max_features_per_image: int = 200):
        """
        Построение словаря визуальных слов методом K-means.
        
        Args:
            train_data: Список кортежей (путь к изображению, класс)
            max_features_per_image: Максимальное количество дескрипторов на изображение
        """
        print("Извлечение дескрипторов из тренировочных изображений...")
        all_descriptors = []
        
        for img_path, _ in train_data:
            try:
                descriptors = self.extract_features(img_path)
                
                if len(descriptors) > 0:
                    # Ограничиваем количество дескрипторов для ускорения
                    if len(descriptors) > max_features_per_image:
                        indices = np.random.choice(len(descriptors), max_features_per_image, replace=False)
                        descriptors = descriptors[indices]
                    
                    all_descriptors.append(descriptors)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
                continue
        
        if len(all_descriptors) == 0:
            raise ValueError("Не удалось извлечь дескрипторы из изображений")
        
        # Объединяем все дескрипторы
        all_descriptors = np.vstack(all_descriptors)
        print(f"Всего дескрипторов: {len(all_descriptors)}")
        
        # Кластеризация методом K-means
        print(f"Построение словаря из {self.vocab_size} слов...")
        kmeans = KMeans(n_clusters=self.vocab_size, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(all_descriptors)
        
        self.vocabulary = kmeans.cluster_centers_
        print("Словарь построен успешно")
    
    def create_histogram(self, img_path: str) -> np.ndarray:
        """
        Создание гистограммы визуальных слов для изображения.
        
        Args:
            img_path: Путь к изображению
            
        Returns:
            Гистограмма размером vocab_size
        """
        if self.vocabulary is None:
            raise ValueError("Словарь не построен. Сначала вызовите build_vocabulary()")
        
        # Извлекаем дескрипторы
        descriptors = self.extract_features(img_path)
        
        if len(descriptors) == 0:
            # Если дескрипторы не найдены, возвращаем нулевую гистограмму
            return np.zeros(self.vocab_size)
        
        # Находим ближайшие центроиды для каждого дескриптора
        # Используем евклидово расстояние
        distances = np.sqrt(((descriptors[:, np.newaxis, :] - self.vocabulary[np.newaxis, :, :]) ** 2).sum(axis=2))
        nearest_centroids = np.argmin(distances, axis=1)
        
        # Создаем гистограмму
        histogram = np.bincount(nearest_centroids, minlength=self.vocab_size)
        
        # Нормализуем гистограмму
        if histogram.sum() > 0:
            histogram = histogram.astype(np.float32) / histogram.sum()
        
        return histogram
    
    def train(self, train_data: List[Tuple[str, int]]):
        """
        Обучение классификатора.
        
        Args:
            train_data: Список кортежей (путь к изображению, класс)
        """
        print("Обучение классификатора 'мешок слов'...")
        
        # Строим словарь, если он еще не построен
        if self.vocabulary is None:
            self.build_vocabulary(train_data)
        
        # Создаем гистограммы для всех тренировочных изображений
        print("Создание гистограмм для тренировочных изображений...")
        histograms = []
        labels = []
        
        for img_path, label in train_data:
            try:
                histogram = self.create_histogram(img_path)
                histograms.append(histogram)
                labels.append(label)
            except Exception as e:
                print(f"Ошибка при создании гистограммы для {img_path}: {e}")
                continue
        
        histograms = np.array(histograms)
        labels = np.array(labels)
        
        print(f"Создано гистограмм: {len(histograms)}")
        
        # Нормализация признаков
        histograms_scaled = self.scaler.fit_transform(histograms)
        
        # Обучение SVM классификатора
        print("Обучение SVM классификатора...")
        self.classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.classifier.fit(histograms_scaled, labels)
        
        self.is_trained = True
        print("Обучение завершено")
    
    def predict(self, img_path: str) -> int:
        """
        Предсказание класса для изображения.
        
        Args:
            img_path: Путь к изображению
            
        Returns:
            Предсказанный класс
        """
        if not self.is_trained:
            raise ValueError("Классификатор не обучен. Сначала вызовите train()")
        
        histogram = self.create_histogram(img_path)
        histogram_scaled = self.scaler.transform(histogram.reshape(1, -1))
        
        prediction = self.classifier.predict(histogram_scaled)[0]
        return int(prediction)
    
    def predict_batch(self, img_paths: List[str]) -> np.ndarray:
        """
        Предсказание классов для списка изображений.
        
        Args:
            img_paths: Список путей к изображениям
            
        Returns:
            Массив предсказанных классов
        """
        if not self.is_trained:
            raise ValueError("Классификатор не обучен. Сначала вызовите train()")
        
        histograms = []
        for img_path in img_paths:
            try:
                histogram = self.create_histogram(img_path)
                histograms.append(histogram)
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
                histograms.append(np.zeros(self.vocab_size))
        
        histograms = np.array(histograms)
        histograms_scaled = self.scaler.transform(histograms)
        
        predictions = self.classifier.predict(histograms_scaled)
        return predictions
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> dict:
        """
        Оценка точности классификатора на тестовой выборке.
        
        Args:
            test_data: Список кортежей (путь к изображению, класс)
            
        Returns:
            Словарь с метриками (accuracy, precision, recall, confusion_matrix)
        """
        if not self.is_trained:
            raise ValueError("Классификатор не обучен. Сначала вызовите train()")
        
        print("Оценка качества на тестовой выборке...")
        
        img_paths = [path for path, _ in test_data]
        true_labels = np.array([label for _, label in test_data])
        
        predictions = self.predict_batch(img_paths)
        
        # Вычисляем метрики
        accuracy = np.mean(predictions == true_labels)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predictions': predictions
        }
        
        return results
    
    def save(self, filepath: str):
        """Сохранение модели в файл."""
        model_data = {
            'vocab_size': self.vocab_size,
            'detector_type': self.detector_type,
            'vocabulary': self.vocabulary,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str):
        """Загрузка модели из файла."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.detector_type = model_data['detector_type']
        self.vocabulary = model_data['vocabulary']
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        
        self._init_detector()
        
        print(f"Модель загружена из {filepath}")
    
    def visualize_keypoints(self, img_path: str, save_path: Optional[str] = None):
        """
        Визуализация ключевых точек на изображении.
        
        Args:
            img_path: Путь к изображению
            save_path: Путь для сохранения изображения (опционально)
        """
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if save_path:
            cv2.imwrite(save_path, img_with_keypoints)
        
        return img_with_keypoints

