import cv2
import numpy as np
import os
import urllib.request
from abc import ABC, abstractmethod

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Скачиваю {os.path.basename(dest_path)}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Готово: {os.path.basename(dest_path)}")
    else:
        print(f"Уже есть: {os.path.basename(dest_path)}")

class BaseDetector(ABC):
    def __init__(self, model_path, config_path, classes_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.classes = self._load_classes(classes_path)
        self.net = self._load_model()
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def _load_classes(self, classes_path):
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
        return classes
    
    def _load_model(self):
        if self.config_path.endswith('.cfg'):
            net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
        elif self.config_path.endswith('.prototxt'):
            net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        else:
            net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)
        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    @abstractmethod
    def preprocess(self, image):
        pass
    
    @abstractmethod
    def postprocess(self, outputs, image_shape):
        pass
    
    def detect(self, image):
        blob = self.preprocess(image)
        self.net.setInput(blob)
        
        layer_names = self.get_output_layers()
        if layer_names:
            outputs = self.net.forward(layer_names)
        else:
            outputs = self.net.forward()
        
        return self.postprocess(outputs, image.shape)
    
    @abstractmethod
    def get_output_layers(self):
        pass