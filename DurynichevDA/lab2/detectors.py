from abc import ABC, abstractmethod
import cv2
import numpy as np


class Detector(ABC):
    @abstractmethod
    def detect(self, image, conf_threshold=0.5, nms_threshold=0.4):
        pass


class SSDDetector(Detector):
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path

        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)

    def detect(self, image, conf_threshold=0.5, nms_threshold=0.4):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=0.007843, size=(300, 300),
            mean=(127.5, 127.5, 127.5), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()  # один выход

        boxes = []
        confidences = []
        class_ids = []

        for detection in output[0, 0]:
            confidence = detection[2]
            if confidence < conf_threshold:
                continue

            class_id = int(detection[1])
            class_name = self.classes[class_id]

            if class_name not in ["car", "bus", "truck", "motorbike"]:
                continue

            x1 = int(detection[3] * w)
            y1 = int(detection[4] * h)
            x2 = int(detection[5] * w)
            y2 = int(detection[6] * h)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        # NMS — работает во всех версиях OpenCV
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        results = []
        if len(indices) > 0:
            if isinstance(indices, tuple):      # OpenCV ≥ 4.5.4
                indices = indices[0]
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]]
                })
        return results


class YOLODetector(Detector):
    def __init__(self, model_path: str, config_path: str, names_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.names_path = names_path

        with open(self.names_path, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
        layer_names = self.net.getLayerNames()
        # Поддержка как старых, так и новых версий OpenCV
        unconnected = self.net.getUnconnectedOutLayers()
        if unconnected.ndim == 1:
            unconnected = unconnected.flatten()
        self.output_layers = [layer_names[i - 1] for i in unconnected]

    def detect(self, image, conf_threshold=0.5, nms_threshold=0.4):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold and self.classes[class_id] in ["car", "bus", "truck", "motorbike"]:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    width = int(detection[2] * w)
                    height = int(detection[3] * h)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS — работает во всех версиях
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        results = []
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]]
                })
        return results