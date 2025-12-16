# src/faster_rcnn_detector.py

import cv2
import numpy as np
from .detector_base import ObjectDetector

class FasterRCNNDetector(ObjectDetector):
    """
    Класс для детектирования объектов с использованием Faster R-CNN.
    """
    def __init__(self, model_path, config_path, classes_path, conf_threshold=0.5, nms_threshold=0.4):
        super().__init__(model_path, config_path, classes_path, conf_threshold, nms_threshold)
        # Загружаем модель
        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        # Для Faster R-CNN нет необходимости явно получать выходные слои

    def preprocess(self, image):
        """
        Предобработка изображения для Faster R-CNN.
        Изменение размера, нормализация.
        """
        height, width = image.shape[:2]
        # Faster R-CNN обычно работает с различными размерами, но часто требует определенного соотношения сторон.
        # Здесь используем размер 640x640, как указано в названии конфига.
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (640, 640)), 1.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        return blob

    def postprocess(self, outputs, original_image_shape):
        """
        Постобработка выхода Faster R-CNN.
        Обработка вывода, фильтрация, NMS.
        """
        height, width = original_image_shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        # Выход Faster R-CNN имеет форму [1, 1, N, 7]
        detections = outputs[0, 0, :, :]

        for i in range(detections.shape[0]):
            confidence = detections[i, 2]
            if confidence > self.conf_threshold:
                class_id = int(detections[i, 1])
                # Координаты в относительных значениях [0,1]
                x1 = int(detections[i, 3] * width)
                y1 = int(detections[i, 4] * height)
                x2 = int(detections[i, 5] * width)
                y2 = int(detections[i, 6] * height)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Подавление немаксимумов (NMS)
        boxes_xywh = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in boxes]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.nms_threshold)

        final_boxes = []
        final_class_ids = []
        final_confidences = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_class_ids.append(class_ids[i])
                final_confidences.append(confidences[i])

        return final_boxes, final_class_ids, final_confidences

    def get_output_layers(self):
        """
        Возвращает имена выходных слоев для Faster R-CNN.
        Для TensorFlow моделей, загруженных через readNetFromTensorflow, этот метод может быть пустым.
        """
        return []
