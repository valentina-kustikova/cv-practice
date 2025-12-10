import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BagOfWords:
    
    def __init__(self, 
                 vocab_size: int = 100,
                 detector_type: str = 'SIFT',
                 descriptor_type: str = 'SIFT'):
        self.vocab_size = vocab_size
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.detector = None
        self.descriptor = None
        self.kmeans = None
        self.classifier = None
        self.scaler = StandardScaler()
        self._init_detector_descriptor()
    
    def _init_detector_descriptor(self):
        if self.detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.descriptor = cv2.SIFT_create()
        elif self.detector_type == 'ORB':
            self.detector = cv2.ORB_create()
            self.descriptor = cv2.ORB_create()
        elif self.detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create()
            self.descriptor = cv2.AKAZE_create()
        else:
            raise ValueError(f"Неподдерживаемый тип детектора: {self.detector_type}")
    
    def extract_features(self, images: List[np.ndarray], max_descriptors_per_image: int = 200) -> np.ndarray:
        descriptors_list = []
        
        for i, img in enumerate(images):
            try:
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                keypoints = self.detector.detect(gray, None)
                if len(keypoints) > 0:
                    if len(keypoints) > max_descriptors_per_image:
                        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_descriptors_per_image]
                    
                    _, descriptors = self.descriptor.compute(gray, keypoints)
                    if descriptors is not None and len(descriptors) > 0:
                        descriptors_list.append(descriptors)
                
                if (i + 1) % 20 == 0:
                    print(f"  Обработано {i + 1}/{len(images)} изображений...")
            except Exception as e:
                print(f"  Ошибка при обработке изображения {i}: {e}")
                continue
        
        if len(descriptors_list) == 0:
            return np.array([])
        
        print(f"  Объединение {len(descriptors_list)} наборов дескрипторов...")
        all_descriptors = descriptors_list[0]
        for desc in descriptors_list[1:]:
            all_descriptors = np.vstack([all_descriptors, desc])
        
        return all_descriptors
    
    def build_vocabulary(self, images: List[np.ndarray], max_samples: int = 3000, max_descriptors_per_image: int = 200):
        print("Извлечение признаков для построения словаря...")
        all_descriptors = self.extract_features(images, max_descriptors_per_image=max_descriptors_per_image)
        
        if len(all_descriptors) == 0:
            raise ValueError("Не удалось извлечь дескрипторы из изображений")
        
        print(f"Извлечено {len(all_descriptors)} дескрипторов")
        
        if len(all_descriptors) > max_samples:
            print(f"Ограничение до {max_samples} дескрипторов для кластеризации...")
            indices = np.random.choice(len(all_descriptors), max_samples, replace=False)
            all_descriptors = all_descriptors[indices]
        
        print(f"Кластеризация {len(all_descriptors)} дескрипторов в {self.vocab_size} кластеров...")
        self.kmeans = KMeans(n_clusters=self.vocab_size, random_state=42, n_init=10, max_iter=300)
        self.kmeans.fit(all_descriptors)
        print("Словарь построен!")
    
    def image_to_histogram(self, image: np.ndarray, max_keypoints: int = 200) -> np.ndarray:
        if self.kmeans is None:
            raise ValueError("Словарь не построен. Сначала вызовите build_vocabulary()")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints = self.detector.detect(gray, None)
        if len(keypoints) == 0:
            return np.zeros(self.vocab_size)
        
        if len(keypoints) > max_keypoints:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_keypoints]
        
        _, descriptors = self.descriptor.compute(gray, keypoints)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size)
        
        labels = self.kmeans.predict(descriptors)
        
        histogram = np.zeros(self.vocab_size)
        for label in labels:
            histogram[label] += 1
        
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def visualize_keypoints(self, image: np.ndarray, output_path: str = None, max_keypoints: int = 50):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vis_image = image.copy()
        else:
            gray = image
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        keypoints = self.detector.detect(gray, None)
        
        if len(keypoints) > max_keypoints:
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_keypoints]
        
        vis_image = cv2.drawKeypoints(vis_image, keypoints, None, 
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"  Визуализация сохранена: {output_path}")
        
        return vis_image
    
    def train(self, images: List[np.ndarray], labels: List[int], visualize: bool = False, vis_dir: str = "visualizations"):
        print("Построение словаря...")
        self.build_vocabulary(images)
        
        if visualize:
            os.makedirs(vis_dir, exist_ok=True)
            print(f"\nСоздание визуализаций ключевых точек в {vis_dir}/...")
            classes_visualized = set()
            for i, (img, label) in enumerate(zip(images, labels)):
                if label not in classes_visualized:
                    vis_path = os.path.join(vis_dir, f"keypoints_class_{label}_img_{i}.jpg")
                    self.visualize_keypoints(img, vis_path, max_keypoints=50)
                    classes_visualized.add(label)
                    if len(classes_visualized) >= 3:
                        break
        
        print("\nПреобразование изображений в гистограммы...")
        histograms = []
        for i, img in enumerate(images):
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(images)} изображений")
            hist = self.image_to_histogram(img)
            histograms.append(hist)
        
        histograms = np.array(histograms)
        
        histograms = self.scaler.fit_transform(histograms)
        
        print("Обучение классификатора SVM...")
        self.classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.classifier.fit(histograms, labels)
        print("Обучение завершено!")
    
    def predict(self, images: List[np.ndarray]) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("Классификатор не обучен. Сначала вызовите train()")
        
        histograms = []
        for img in images:
            hist = self.image_to_histogram(img)
            histograms.append(hist)
        
        histograms = np.array(histograms)
        histograms = self.scaler.transform(histograms)
        
        predictions = self.classifier.predict(histograms)
        return predictions
    
    def evaluate(self, images: List[np.ndarray], labels: List[int]) -> float:
        predictions = self.predict(images)
        accuracy = accuracy_score(labels, predictions)
        return accuracy
    
    def save(self, filepath: str):
        model_data = {
            'vocab_size': self.vocab_size,
            'detector_type': self.detector_type,
            'descriptor_type': self.descriptor_type,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.detector_type = model_data['detector_type']
        self.descriptor_type = model_data['descriptor_type']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self._init_detector_descriptor()
        print(f"Модель загружена из {filepath}")
