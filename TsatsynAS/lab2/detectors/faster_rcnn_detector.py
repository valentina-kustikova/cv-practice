import cv2
import numpy as np
from .base_detector import BaseDetector


class FasterRCNNDetector(BaseDetector):
    def __init__(self, model_path, config_path, classes_path, conf_threshold=0.5):
        self.classes_path = classes_path
        # Загружаем классы в другое поле, чтобы избежать конфликта
        self._classes = self._load_classes()
        print(f"FasterRCNN CONSTRUCTOR: Loaded {len(self._classes)} classes")
        print(f"FasterRCNN CONSTRUCTOR: Class 2 is: '{self._classes[2]}'")

        # Теперь вызываем родительский конструктор
        super().__init__(model_path, config_path, conf_threshold)

    def _load_classes(self):
        try:
            with open(self.classes_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            return classes
        except Exception as e:
            print(f"Error loading classes from {self.classes_path}: {e}")
            # Fallback to default COCO classes
            return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def load_model(self):
        self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(300, 300),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        return blob

    def postprocess(self, outputs, image_shape):
        height, width = image_shape[:2]
        results = []

        print(f"Faster R-CNN DEBUG: Output shape: {outputs[0].shape}")
        print(f"Faster R-CNN DEBUG: Classes available: {len(self._classes)}")

        detections = outputs[0][0, 0]  # Берем [batch, detection] → форма (100, 7)

        for i in range(min(10, detections.shape[0])):  # Проверим только первые 10
            detection = detections[i]
            class_id = int(detection[1])
            confidence = float(detection[2])

            # Используем self._classes вместо self.classes
            if class_id < len(self._classes):
                class_name = self._classes[class_id]
                is_valid_class = True
            else:
                class_name = f"OUT_OF_RANGE({class_id})"
                is_valid_class = False

            print(f"Faster R-CNN DEBUG: Detection {i}: class={class_id}({class_name}), conf={confidence:.3f}")

            # Принимаем детекции с confidence > threshold и валидным классом
            if confidence > self.conf_threshold and is_valid_class:
                x1 = float(detection[3]) * width
                y1 = float(detection[4]) * height
                x2 = float(detection[5]) * width
                y2 = float(detection[6]) * height

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Минимальная проверка координат
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0:
                    results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })

        print(f"Faster R-CNN DEBUG: Total results: {len(results)}")
        return results
