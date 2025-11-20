import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os
from .utils import extract_class_name


class BagOfWordsClassifier:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.detector = cv2.SIFT_create()
        self.kmeans = KMeans(n_clusters=vocab_size, random_state=42)
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ])
        self.vocab = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}

    def extract_features(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        kp, descriptors = self.detector.detectAndCompute(img, None)
        if descriptors is None:
            # Return empty array if no features detected
            return np.array([])

        return descriptors

    def build_vocabulary(self, image_paths):
        all_descriptors = []

        for img_path in image_paths:
            descriptors = self.extract_features(img_path)
            if len(descriptors) > 0:
                all_descriptors.append(descriptors)

        if len(all_descriptors) == 0:
            raise ValueError("No features could be extracted from any image")

        all_descriptors = np.vstack(all_descriptors)

        self.kmeans.fit(all_descriptors)
        self.vocab = self.kmeans.cluster_centers_

        print(f"Vocabulary built with {len(self.vocab)} words")

    def create_histogram(self, image_path):
        descriptors = self.extract_features(image_path)

        if len(descriptors) == 0:
            # Return zero histogram if no features
            return np.zeros(self.vocab_size)

        labels = self.kmeans.predict(descriptors)

        hist, _ = np.histogram(labels, bins=np.arange(self.vocab_size + 1), density=True)
        return hist

    def prepare_data(self, image_paths):
        histograms = []
        labels = []

        for img_path in image_paths:
            hist = self.create_histogram(img_path)
            histograms.append(hist)

            class_name = extract_class_name(img_path)
            if class_name not in self.label_encoder:
                idx = len(self.label_encoder)
                self.label_encoder[class_name] = idx
                self.reverse_label_encoder[idx] = class_name
            labels.append(self.label_encoder[class_name])

        return np.array(histograms), np.array(labels)

    def train(self, train_paths):
        print("Building vocabulary...")
        self.build_vocabulary(train_paths)

        print("Preparing training data...")
        X_train, y_train = self.prepare_data(train_paths)

        print("Training classifier...")
        self.classifier.fit(X_train, y_train)

        print("Training completed.")

    def predict(self, image_path):
        hist = self.create_histogram(image_path)
        hist = hist.reshape(1, -1)

        pred = self.classifier.predict(hist)[0]
        prob = self.classifier.predict_proba(hist)[0]

        class_name = self.reverse_label_encoder[pred]
        confidence = np.max(prob)

        return class_name, confidence

    def evaluate(self, test_paths):
        X_test, y_test = self.prepare_data(test_paths)

        predictions = self.classifier.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        return accuracy

    def save_model(self, filepath):
        model_data = {
            'vocab': self.vocab,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder,
            'vocab_size': self.vocab_size
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.vocab = model_data['vocab']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.reverse_label_encoder = model_data['reverse_label_encoder']
        self.vocab_size = model_data['vocab_size']

        print(f"Model loaded from {filepath}")
