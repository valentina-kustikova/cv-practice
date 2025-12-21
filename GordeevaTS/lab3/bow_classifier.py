import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class BOWClassifier:
    def __init__(self, detector_type='SIFT', vocab_size=50):
        self.detector_type = detector_type
        self.vocab_size = vocab_size
        self.detector = self._create_detector()
        self.kmeans = None
        self.classifier = None
        self.is_trained = False
        
    def _create_detector(self):
        return cv2.SIFT_create()

    
    def extract_features(self, image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            return descriptors
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def build_vocabulary(self, images, sample_size=50000):
        print("Building visual vocabulary...")
        
        all_descriptors = []
        for i, image in enumerate(images):
            descriptors = self.extract_features(image)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(images)} images for vocabulary")
        
        if not all_descriptors:
            raise ValueError("No features extracted from images")
        
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors: {len(all_descriptors)}")
        
        if len(all_descriptors) > sample_size:
            indices = np.random.choice(len(all_descriptors), sample_size, replace=False)
            all_descriptors = all_descriptors[indices]
            print(f"Sampled to {len(all_descriptors)} descriptors")
        
        actual_vocab_size = min(self.vocab_size, len(all_descriptors) // 10)
        if actual_vocab_size < self.vocab_size:
            print(f"Reducing vocabulary size from {self.vocab_size} to {actual_vocab_size}")
            self.vocab_size = actual_vocab_size
        
        if self.vocab_size < 2:
            raise ValueError("Vocabulary size too small after adjustment")
        
        print(f"Performing K-means with {self.vocab_size} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.vocab_size, 
            random_state=42,
            n_init=10,
            max_iter=100
        )
        self.kmeans.fit(all_descriptors)
        print(f"Vocabulary built with {self.vocab_size} words")
    
    def image_to_bow(self, image):
        descriptors = self.extract_features(image)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size)
        
        visual_words = self.kmeans.predict(descriptors)
        bow_vector = np.bincount(visual_words, minlength=self.vocab_size)
        bow_vector = bow_vector.astype(np.float32)
        if np.sum(bow_vector) > 0:
            bow_vector /= np.sum(bow_vector)
        
        return bow_vector
    
    def train(self, images, labels):
        self.build_vocabulary(images)
        print("Extracting BoW features...")
        bow_features = []
        for i, image in enumerate(images):
            bow_vector = self.image_to_bow(image)
            bow_features.append(bow_vector)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(images)} images")
        
        bow_features = np.array(bow_features)
        print(f"BoW features shape: {bow_features.shape}")
        
        print("Training classifier...")
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ])
        
        self.classifier.fit(bow_features, labels)
        self.is_trained = True
        print("Training completed")
    
    def predict(self, images):
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")
        
        bow_features = []
        for i, image in enumerate(images):
            bow_vector = self.image_to_bow(image)
            bow_features.append(bow_vector)
            
            if (i + 1) % 20 == 0:
                print(f"Predicting {i + 1}/{len(images)} images")
        
        bow_features = np.array(bow_features)
        return self.classifier.predict(bow_features)
    
    def save(self, path):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'detector_type': self.detector_type,
            'vocab_size': self.vocab_size,
            'kmeans': self.kmeans,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved successfully to {path}")
    
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.detector_type = model_data['detector_type']
        self.vocab_size = model_data['vocab_size']
        self.kmeans = model_data['kmeans']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        self.detector = self._create_detector()
        
        print(f"Model loaded successfully from {path}")
        print(f"Detector: {self.detector_type}, Vocab size: {self.vocab_size}")
    
    def get_model_info(self):
        return {
            'algorithm': 'BOW',
            'detector': self.detector_type,
            'vocab_size': self.vocab_size,
            'is_trained': self.is_trained
        }