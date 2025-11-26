import cv2
import numpy as np
from .base_detector import BaseDetector

class SSDDetector(BaseDetector):
    """Детектор на основе SSD"""
    
    def __init__(self, model_path, classes_file, confidence_threshold=0.5, nms_threshold=0.4):
        super().__init__(model_path, None, classes_file, confidence_threshold, nms_threshold)
    
    def preprocess(self, image):
        """Предобработка изображения для SSD"""
        original_shape = image.shape[:2]
        
        # Создание blob для SSD
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,  # Масштабирование
            size=(300, 300),  # Размер входного изображения
            mean=(127.5, 127.5, 127.5),  # Средние значения
            swapRB=True,  # Конвертация BGR to RGB
            crop=False
        )
        
        return blob, original_shape
    
    def postprocess(self, outputs, image_shape):
        """Постобработка выходов SSD"""
        height, width = image_shape
        detections = outputs[0][0]  # Первый выход содержит детекции
        
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in detections:
            confidence = detection[2]
            
            if confidence > self.confidence_threshold:
                class_id = int(detection[1])
                
                # Координаты bounding box (нормализованные)
                x1 = int(detection[3] * width)
                y1 = int(detection[4] * height)
                x2 = int(detection[5] * width)
                y2 = int(detection[6] * height)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        # Применение Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else str(class_ids[i]),
                    'confidence': confidences[i],
                    'bbox': (x, y, x + w, y + h)
                })
        
        return results