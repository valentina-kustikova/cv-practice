import numpy as np
import cv2
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class SimpleCNN:
    def __init__(self, input_shape=(64, 64, 3)):
        self.input_shape = input_shape
        self.filters = []
        
    def extract_features(self, images):
        features = []
        
        for image in images:
            resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            
            feature_vector = []
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            hog_features = self.calculate_hog(gray)
            
            lbp_features = self.calculate_lbp(gray)
            
            color_features = []
            for channel in range(3):
                channel_data = resized[:, :, channel]
                color_features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data)
                ])
            
            feature_vector.extend(hog_features)
            feature_vector.extend(lbp_features)
            feature_vector.extend(color_features)
            feature_vector.extend([np.mean(sobelx), np.std(sobelx), np.mean(sobely), np.std(sobely)])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def calculate_hog(self, image, cell_size=8, block_size=2, nbins=9):
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        h, w = image.shape
        features = []
        
        for i in range(0, h - cell_size + 1, cell_size):
            for j in range(0, w - cell_size + 1, cell_size):
                cell_magnitude = magnitude[i:i+cell_size, j:j+cell_size]
                cell_orientation = orientation[i:i+cell_size, j:j+cell_size]
                
                hist, _ = np.histogram(cell_orientation, bins=nbins, range=(0, 180), weights=cell_magnitude)
                features.extend(hist)
        
        features = np.array(features)
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
        
        return features
    
    def calculate_lbp(self, image, radius=1, points=8):
        h, w = image.shape
        lbp_image = np.zeros_like(image)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary = ''
                for p in range(points):
                    angle = 2 * np.pi * p / points
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)
                    x = int(round(x))
                    y = int(round(y))
                    binary += '1' if image[x, y] >= center else '0'
                
                lbp_image[i, j] = int(binary, 2)
        
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        if np.sum(hist) > 0:
            hist /= np.sum(hist)
        
        return hist

class NNClassifier:
    def __init__(self, classifier_type='RandomForest', num_classes=3, input_shape=(64, 64, 3)):
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.feature_extractor = SimpleCNN(input_shape)
        self.classifier = None
        self.is_trained = False
        
    def _create_classifier(self):
        if self.classifier_type == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def train(self, images, labels):
        print("Извлечение признаков...")
        features = self.feature_extractor.extract_features(images)
        print(f"Извлечено {features.shape[1]} признаков из {features.shape[0]} изображений")
        
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self._create_classifier())
        ])
        
        print("Обучение классификатора...")
        self.classifier.fit(features, labels)
        self.is_trained = True
        print("Обучение завершено")
    
    def predict(self, images):
        if not self.is_trained:
            raise ValueError("Классификатор не обучен")
        
        features = self.feature_extractor.extract_features(images)
        return self.classifier.predict(features)
    
    def save(self, path):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'classifier_type': self.classifier_type,
            'classifier': self.classifier,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
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
        
        self.classifier_type = model_data['classifier_type']
        self.classifier = model_data['classifier']
        self.input_shape = model_data['input_shape']
        self.num_classes = model_data['num_classes']
        self.is_trained = model_data['is_trained']
        self.feature_extractor = SimpleCNN(self.input_shape)
        
        print(f"Model loaded successfully from {path}")
        print(f"Classifier: {self.classifier_type}")
    
    def get_model_info(self):
        return {
            'algorithm': 'NN',
            'classifier': self.classifier_type,
            'input_shape': self.input_shape,
            'is_trained': self.is_trained
        }
