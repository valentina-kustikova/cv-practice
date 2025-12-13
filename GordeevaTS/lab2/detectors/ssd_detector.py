import cv2
import numpy as np
import os
from .base_detector import BaseDetector

class SSDMobileNetDetector(BaseDetector):
    def __init__(self, model_path, config_path, classes_path, confidence_threshold=0.5):
        self._check_files_exist(model_path, config_path, classes_path)
        
        super().__init__(model_path, config_path, classes_path, confidence_threshold)
        
        self.ssd_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
            'sofa', 'train', 'tvmonitor'
        ]
    
    def _check_files_exist(self, model_path, config_path, classes_path):
        missing_files = []
        
        if not os.path.exists(model_path):
            missing_files.append(f"Модель: {model_path}")
        if not os.path.exists(config_path):
            missing_files.append(f"Конфиг: {config_path}")
        if not os.path.exists(classes_path):
            missing_files.append(f"Классы: {classes_path}")
        
        if missing_files:
            print("Ошибка: Отсутствуют файлы модели:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nЗапустите скрипт для скачивания моделей:")
            print("  python download_models.py")
            raise FileNotFoundError(f"Отсутствуют файлы модели: {missing_files}")
    
    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0/127.5,
            size=(300, 300),        
            mean=(127.5, 127.5, 127.5),  
            swapRB=False,           
            crop=False
        )
        return blob
    
    def postprocess(self, outputs, image_shape):
        height, width = image_shape[:2]
        detections = []

        if len(outputs) > 0:
            detections_tensor = outputs[0][0]  

            for detection in detections_tensor:
                confidence = float(detection[2])

                if confidence < self.confidence_threshold:
                    continue

                class_id = int(detection[1])
                
                if class_id <= 0 or class_id >= len(self.ssd_classes):
                    continue
                
                class_name = self.ssd_classes[class_id].lower()

                x1_norm = detection[3]
                y1_norm = detection[4]
                x2_norm = detection[5]
                y2_norm = detection[6]

                x1 = int(x1_norm * width)
                y1 = int(y1_norm * height)
                x2 = int(x2_norm * width)
                y2 = int(y2_norm * height)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                if x2 <= x1:
                    continue  
                if y2 <= y1:
                    continue

                if class_name == 'motorbike':
                    class_name = 'motorcycle'
                elif class_name == 'aeroplane':
                    class_name = 'airplane' 
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

        return detections
    
    def get_output_layers(self):
        return []