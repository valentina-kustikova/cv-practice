import cv2
import numpy as np
import os
from abc import ABC, abstractmethod


# ============================================================================
# Иерархия классов детекторов объектов (Model Layer):
# - BaseDetector: абстрактный базовый класс, определяющий интерфейс (preprocess, postprocess)
# - SSDDetector: реализация детектора на базе MobileNet SSD (Caffe)
# - YOLODetector: реализация детектора на базе YOLOv3 (Darknet)
# Отвечает за загрузку весов нейросетей и инференс (получение предсказаний)
# ============================================================================

class BaseDetector(ABC):
    def __init__(self, config_path, weights_path, classes_path=None, conf_threshold=0.5, nms_threshold=0.4):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = []

        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

        # Генерация случайных цветов для классов
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    @abstractmethod
    def preprocess(self, image):
        """Подготовка изображения для подачи в сеть (ресайз, нормализация)"""
        pass

    @abstractmethod
    def postprocess(self, image, outputs):
        """Обработка сырого выхода нейросети в список боксов"""
        pass

    def detect(self, image):
        """Основной метод: предобработка -> прогон сети -> постобработка"""
        blob = self.preprocess(image)
        self.net.setInput(blob)
        output_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_names)
        return self.postprocess(image, outputs)

    def get_color(self, label_str):
        """Возвращает уникальный цвет для строки класса"""
        idx = hash(label_str) % 100
        return [int(c) for c in self.colors[idx]]


class SSDDetector(BaseDetector):
    def __init__(self, config_path, weights_path, classes_path=None):
        super().__init__(config_path, weights_path, classes_path)
        if not self.classes:
            self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"]

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    def postprocess(self, image, outputs):
        (h, w) = image.shape[:2]
        detections = outputs[0]
        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])

                # В VOC: bus=6, car=7
                if idx not in [6, 7]:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label_str = self.classes[idx]
                results.append([label_str, confidence, startX, startY, endX - startX, endY - startY])

        return results


class YOLODetector(BaseDetector):
    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    def postprocess(self, image, outputs):
        (H, W) = image.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.conf_threshold:

                    # В COCO: car=2, bus=5
                    if classID not in [2, 5]:
                        continue

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                label_str = self.classes[classIDs[i]]
                results.append([label_str, confidences[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])

        return results
