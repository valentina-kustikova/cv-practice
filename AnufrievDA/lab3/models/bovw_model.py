import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import List, Tuple

class BoVWClassifier:
    def __init__(self, n_clusters: int = 100, detector_name: str = 'SIFT'):
        self.n_clusters = n_clusters
        self.detector_name = detector_name
        
        # ИЗМЕНЕНИЕ 1: Теперь у нас один объект "extractor", который делает всё
        self.extractor = self._init_feature_extractor()
        
        # Используем MiniBatchKMeans
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
        
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
        ])
        self.is_fitted = False

    def _init_feature_extractor(self):
        if self.detector_name == 'SIFT':
            return cv2.SIFT_create(nfeatures=500)
        elif self.detector_name == 'ORB':
            return cv2.ORB_create(nfeatures=500)
        else:
            raise ValueError(f"Unknown detector: {self.detector_name}")

    def _extract_features(self, image_paths: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        all_descriptors = []
        per_img_desc = []

        print(f"BoVW: Извлечение признаков ({self.detector_name})...")
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    per_img_desc.append(None)
                    continue
            
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                kp, des = self.extractor.detectAndCompute(gray, None)

                if des is not None and len(des) > 0:
                    des = des.astype(np.float64)
                    all_descriptors.append(des)
                    per_img_desc.append(des)
                else:
                    per_img_desc.append(None)

            except Exception as e:
                print(f"Ошибка {path}: {e}")
                per_img_desc.append(None)

        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors)
        else:
            all_descriptors = np.array([])

        return all_descriptors, per_img_desc
        
    def fit(self, image_paths: List[str], labels: List[int]):
        all_desc, des_list = self._extract_features(image_paths)
        if len(all_desc) == 0: raise ValueError("Нет дескрипторов!")

        print(f"BoVW: обучение KMeans ({len(all_desc)} векторов)...")
        self.kmeans.fit(all_desc)

        print("BoVW: построение гистограмм...")
        X_hist = self._images_to_histograms(des_list)
        print(f"BoVW: обучение SVM...")
        self.classifier.fit(X_hist, labels)
        self.is_fitted = True
        print("BoVW обучен.")

    def _images_to_histograms(self, descriptors_list: List[np.ndarray]) -> np.ndarray:
        histograms = []
        for des in descriptors_list:
            hist = np.zeros(self.n_clusters)
            if des is not None and len(des) > 0:
                words = self.kmeans.predict(des.astype(np.float64))
                for w in words: hist[w] += 1
                if np.sum(hist) > 0: hist = hist / np.sum(hist)
            histograms.append(hist)
        return np.array(histograms)

    def predict(self, image_paths: List[str]) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Model not fitted")
        _, des_list = self._extract_features(image_paths)
        X_hist = self._images_to_histograms(des_list)
        return self.classifier.predict(X_hist)

    def save(self, path: str):
        # Сохраняем параметры, игнорируя объект OpenCV
        data = {
            'n_clusters': self.n_clusters,
            'detector_name': self.detector_name,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, path)
        print(f"BoVW сохранён в: {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        # Создаем объект (extractor создастся внутри __init__)
        model = cls(n_clusters=data['n_clusters'], detector_name=data['detector_name'])
        model.kmeans = data['kmeans']
        model.classifier = data['classifier']
        model.is_fitted = data['is_fitted']
        return model
        
    def visualize_keypoints(self, image_path: str, save_to: str = "keypoints.jpg"):
        img = cv2.imread(image_path)
        if img is None: return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ИЗМЕНЕНИЕ 4: Используем self.extractor
        kp, _ = self.extractor.detectAndCompute(gray, None)
        
        out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(save_to, out)
        print(f"Визуализация сохранена: {save_to}")