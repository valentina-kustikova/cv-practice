import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import List, Tuple
import os

class BoVWClassifier:
    def __init__(self, n_clusters: int = 100, detector_name: str = 'SIFT'):
        self.n_clusters = n_clusters
        self.detector_name = detector_name
        self.detector, self.descriptor = self._init_feature_extractor()
        
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, n_init=3)
        
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
        ])
        self.is_fitted = False

    def _init_feature_extractor(self):
        if self.detector_name == 'SIFT':
            sift = cv2.SIFT_create(nfeatures=500)
            return sift, sift
        elif self.detector_name == 'ORB':
            orb = cv2.ORB_create(nfeatures=500)
            return orb, orb
        elif self.detector_name == 'AKAZE':
            akaze = cv2.AKAZE_create()
            return akaze, akaze
        else:
            raise ValueError(f"Unknown detector: {self.detector_name}")

    def _extract_features(self, image_paths: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        all_descriptors = []
        per_img_desc = []

        print(f"BoVW: Извлечение признаков ({self.detector_name})...")
        for i, path in enumerate(image_paths):
            try:
                img = cv2.imread(path)
                if img is None:
                    per_img_desc.append(None)
                    continue

                h, w = img.shape[:2]
                max_dim = 640
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                kp, des = self.descriptor.detectAndCompute(gray, None)

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
        all_des, des_list = self._extract_features(image_paths)
        
        if len(all_des) == 0:
            raise ValueError("Не найдено дескрипторов.")

        print(f"Всего найдено дескрипторов: {len(all_des)}")
        print(f"Обучение кластеризатора...")
        self.kmeans.fit(all_des.astype(np.float64))
        
        print("Построение гистограмм и обучение SVM...")
        X_train = self._images_to_histograms(des_list)
        y_train = np.array(labels)
        
        self.classifier.fit(X_train, y_train)
        self.is_fitted = True
        print("Готово.")

    def _images_to_histograms(self, descriptors_list: List[np.ndarray]) -> np.ndarray:
        histograms = []
        for des in descriptors_list:
            hist = np.zeros(self.n_clusters)
            if des is not None and len(des) > 0:
                predictions = self.kmeans.predict(des.astype(np.float64))
                for pred in predictions:
                    hist[pred] += 1
                if np.sum(hist) > 0:
                    hist = hist / np.sum(hist)
            histograms.append(hist)
        return np.array(histograms)

    def predict(self, image_paths: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        _, des_list = self._extract_features(image_paths)
        X = self._images_to_histograms(des_list)
        return self.classifier.predict(X)

    def visualize_keypoints(self, image_path: str, save_to: str = "keypoints.jpg", max_points: int = 500):
        img = cv2.imread(image_path)
        if img is None: return
        h, w = img.shape[:2]
        if h > 1000 or w > 1000:
            scale = 1000 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, _ = self.detector.detectAndCompute(gray, None)
        kps = sorted(kps, key=lambda x: x.response, reverse=True)[:max_points]
        out_img = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(save_to, out_img)

    # --- МЕТОДЫ СОХРАНЕНИЯ И ЗАГРУЗКИ ---
    def save(self, path: str):
        data = {
            'detector_name': self.detector_name,
            'n_clusters': self.n_clusters,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, path)
        print(f"BoVW model saved to {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(n_clusters=data['n_clusters'], detector_name=data['detector_name'])
        obj.kmeans = data['kmeans']
        obj.classifier = data['classifier']
        obj.is_fitted = data['is_fitted']
        print(f"BoVW model loaded from {path}")
        return obj