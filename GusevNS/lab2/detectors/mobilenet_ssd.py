from pathlib import Path
import cv2
import numpy as np
from .base import BaseDetector, DetectionResult


class MobilenetSsdDetector(BaseDetector):
    CLASSES_VOC = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    def __init__(self, model_dir):
        super().__init__(
            model_dir=model_dir,
            score_threshold=0.2,
            nms_threshold=0.45,
            class_filter=["car", "bus", "motorbike"],
        )

    def load_class_names(self):
        return self.CLASSES_VOC

    def ensure_model_files(self):
        caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
        prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        self.caffemodel_path = self.model_dir / "MobileNetSSD.caffemodel"
        self.prototxt_path = self.model_dir / "MobileNetSSD.prototxt"
        self.download_file(caffemodel_url, self.caffemodel_path)
        self.download_file(prototxt_url, self.prototxt_path)

    def load_network(self):
        self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.caffemodel_path))

    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        return blob

    def infer(self, blob):
        self.net.setInput(blob)
        return self.net.forward()

    def postprocess(self, outputs, original_shape):
        h, w = original_shape
        results = []
        detections = outputs[0, 0]
        for detection in detections:
            confidence = float(detection[2])
            if confidence < self.score_threshold:
                continue
            class_id = int(detection[1])
            if class_id <= 0 or class_id >= len(self.class_names):
                continue
            label = self.class_names[class_id]
            if label not in self.class_filter:
                continue
            x1 = int(detection[3] * w)
            y1 = int(detection[4] * h)
            x2 = int(detection[5] * w)
            y2 = int(detection[6] * h)
            box = (x1, y1, x2 - x1, y2 - y1)
            results.append(DetectionResult(class_id, label, confidence, box))
        return results

    def detect(self, image):
        blob = self.preprocess(image)
        outputs = self.infer(blob)
        return self.postprocess(outputs, image.shape[:2])

