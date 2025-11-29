import cv2
import numpy as np
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Базовый класс для детекторов объектов"""
    
    def __init__(self, model_path, config_path=None, classes_file=None, confidence_threshold=0.5, nms_threshold=0.4):
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.classes = self._load_classes(classes_file)
        self.net = self._load_model()
        self.output_layers = self._get_output_layers()
    
    def _load_classes(self, classes_file):
        """Загрузка классов объектов"""
        if classes_file:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            return classes
        return []
    
    def _load_model(self):
        """Загрузка модели"""
        if self.config_path:
            net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
        else:
            net = cv2.dnn.readNetFromTensorflow(self.model_path)
        return net
    
    def _get_output_layers(self):
        """Получение выходных слоев сети"""
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    @abstractmethod
    def preprocess(self, image):
        """Предобработка изображения"""
        pass
    
    @abstractmethod
    def postprocess(self, outputs, image_shape):
        """Постобработка выходов сети"""
        pass
    
    def detect(self, image):
        """Детектирование объектов на изображении"""
        # Предобработка
        blob, original_shape = self.preprocess(image)
        
        # Прямой проход через сеть
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Постобработка
        detections = self.postprocess(outputs, original_shape)
        
        return detections

    def set_backend(self, backend=cv2.dnn.DNN_BACKEND_OPENCV, target=cv2.dnn.DNN_TARGET_CPU):
        """Установка бэкенда для вычислений"""
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)