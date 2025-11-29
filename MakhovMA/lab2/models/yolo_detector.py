import cv2
import numpy as np
from .base_detector import BaseDetector

class YOLODetector(BaseDetector):
    """Детектор на основе YOLO"""
    
    def __init__(self, model_path, config_path, classes_file, confidence_threshold=0.5, nms_threshold=0.4):
        super().__init__(model_path, config_path, classes_file, confidence_threshold, nms_threshold)
    
    def preprocess(self, image):
        """Предобработка изображения для YOLO"""
        original_shape = image.shape[:2]
        
        # Создание blob с нормализацией и изменением размера
        blob = cv2.dnn.blobFromImage(
            image, 
            1/255.0,  # Масштабирование
            (416, 416),  # Размер входного изображения
            swapRB=True,  # Конвертация BGR to RGB
            crop=False
        )
        
        return blob, original_shape
    
    def postprocess(self, outputs, image_shape):
        """Постобработка выходов YOLO"""
        height, width = image_shape
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Масштабирование координат bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Координаты левого верхнего угла
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Применение Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else str(class_ids[i]),
                    'confidence': confidences[i],
                    'bbox': (x, y, x + w, y + h)
                })
        
        return detections