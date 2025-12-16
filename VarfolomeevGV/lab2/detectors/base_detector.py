"""
Базовый абстрактный класс для детекторов объектов.
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List, Tuple, Optional


class BaseDetector(ABC):
    """Абстрактный базовый класс для всех детекторов объектов."""
    
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Инициализация детектора.
        
        Args:
            conf_threshold: Порог уверенности для фильтрации детекций
            nms_threshold: Порог IoU для Non-Maximum Suppression
        """
        self.net = None
        self.class_names = []
        self.input_size = (640, 640)  # Будет переопределено в подклассах
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_loaded = False
    
    @abstractmethod
    def load_model(self, weights_path: str, config_path: Optional[str] = None, 
                   names_path: Optional[str] = None) -> None:
        """
        Загрузка модели детектирования.
        
        Args:
            weights_path: Путь к файлу весов модели
            config_path: Путь к конфигурационному файлу (если требуется)
            names_path: Путь к файлу с названиями классов
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Предобработка изображения перед подачей в сеть.
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            blob: Подготовленный blob для сети
            metadata: Словарь с метаданными (scale, pad и т.д.) для постобработки
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs, metadata: dict, 
                   original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка выходов сети.
        
        Args:
            outputs: Выходные тензоры сети
            metadata: Метаданные из preprocess
            original_shape: Исходный размер изображения (height, width)
            
        Returns:
            Список детекций в формате [x1, y1, x2, y2, class_id, confidence]
        """
        pass
    
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Основной метод детектирования объектов на изображении.
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            Список детекций в формате [x1, y1, x2, y2, class_id, confidence]
        """
        if not self.model_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")
        
        original_shape = image.shape[:2]  # (height, width)
        
        # Предобработка
        blob, metadata = self.preprocess(image)
        
        # Инференс
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # Постобработка
        detections = self.postprocess(outputs, metadata, original_shape)
        
        return detections
    
    def filter_vehicle_classes(self, detections: List[List[float]]) -> List[List[float]]:
        """
        Фильтрация детекций по классам транспортных средств.
        
        COCO классы транспортных средств (0-based index):
        - 2: car (автомобиль)
        - 5: bus (автобус)
        
        Args:
            detections: Список детекций
            
        Returns:
            Отфильтрованный список детекций
        """
        vehicle_classes = {2, 5}
        filtered = []
        for det in detections:
            if len(det) >= 5 and int(det[4]) in vehicle_classes:  # class_id в позиции 4
                filtered.append(det)
        return filtered

