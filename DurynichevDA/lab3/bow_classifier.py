import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from detectors import SIFTDetector, ORBDetector, AKAZEDetector
from abc import ABC, abstractmethod
from typing import List, Tuple
import pickle
import os
from tqdm import tqdm


class AbstractClassifier(ABC):
    @abstractmethod
    def fit(self, train_data: List[Tuple[str, int]]):
        pass

    @abstractmethod
    def predict(self, test_data: List[Tuple[str, int]]):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        pass


class BoWClassifier(AbstractClassifier):
    def __init__(self, k=600, detector_type="sift"):
        self.k = k
        self.detector_type = detector_type.lower()

        if self.detector_type == "sift":
            self.detector = SIFTDetector()
        elif self.detector_type == "orb":
            self.detector = ORBDetector()
        elif self.detector_type == "akaze":
            self.detector = AKAZEDetector()
        else:
            raise ValueError("detector must be sift/orb/akaze")

        self.kmeans = None
        self.scaler = StandardScaler()
        self.clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)

    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detect_and_compute(gray)
        return des, (img, kp) if kp is not None else (img, [])

    def fit(self, train_data):
        print(f"\nОбучение BoW [{self.detector.name}]")
        all_descs = []
        labels = []

        for path, label in tqdm(train_data, desc="Дескрипторы"):
            des, _ = self.extract_features(path)
            if des is not None and len(des) > 0:
                all_descs.append(des)
                labels.append(label)

        all_descs = np.vstack(all_descs)
        print(f"Кластеризация → {self.k} слов")
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=42)
        self.kmeans.fit(all_descs)

        hists = []
        for path, _ in tqdm(train_data, desc="Гистограммы"):
            des, _ = self.extract_features(path)
            hist = np.zeros(self.k)
            if des is not None:
                words = self.kmeans.predict(des)
                hist, _ = np.histogram(words, bins=self.k, range=(0, self.k), density=True)
            hists.append(hist)

        X = self.scaler.fit_transform(hists)
        self.clf.fit(X, labels)
        print("BoW обучен!")

    def predict(self, test_data):
        print(f"\nТест BoW [{self.detector.name}]")
        hists, y_true = [], []
        for path, label in tqdm(test_data, desc="Загрузка"):
            des, _ = self.extract_features(path)
            hist = np.zeros(self.k)
            if des is not None:
                words = self.kmeans.predict(des)
                hist, _ = np.histogram(words, bins=self.k, range=(0, self.k), density=True)
            hists.append(hist)
            y_true.append(label)

        X = self.scaler.transform(hists)
        y_pred = self.clf.predict(X)
        acc = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred,
              target_names=['Кремль', 'Собор', 'Дворец труда']))
        return acc

    def save(self, path):
        os.makedirs("models", exist_ok=True)
        pickle.dump({
            'kmeans': self.kmeans, 'scaler': self.scaler, 'clf': self.clf,
            'k': self.k, 'detector_type': self.detector_type
        }, open(path, 'wb'))
        print(f"Сохранено → {path}")

    @classmethod
    def load(cls, path):
        data = pickle.load(open(path, 'rb'))
        model = cls(k=data['k'], detector_type=data['detector_type'])
        model.kmeans = data['kmeans']
        model.scaler = data['scaler']
        model.clf = data['clf']
        return model