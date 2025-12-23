from abc import ABC, abstractmethod
from pathlib import Path
import urllib.request
import cv2
import numpy as np


class DetectionResult:
    def __init__(self, class_id, label, confidence, box):
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.box = box


class BaseDetector(ABC):
    def __init__(self, model_dir, score_threshold, nms_threshold, class_filter=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.class_filter = class_filter or []
        self.net = None
        self.class_names = self.load_class_names()
        self.ensure_model_files()
        self.load_network()

    @abstractmethod
    def load_class_names(self):
        return []

    @abstractmethod
    def ensure_model_files(self):
        ...

    @abstractmethod
    def load_network(self):
        ...

    @abstractmethod
    def preprocess(self, image):
        ...

    @abstractmethod
    def infer(self, blob):
        ...

    @abstractmethod
    def postprocess(self, outputs, original_shape):
        ...

    def detect(self, image):
        blob = self.preprocess(image)
        outputs = self.infer(blob)
        return self.postprocess(outputs, image.shape[:2])

    def download_file(self, url, destination):
        destination = Path(destination)
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response, open(destination, "wb") as target:
            target.write(response.read())
        return destination

    def get_label(self, class_id):
        if class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def filter_by_vehicle(self, class_id):
        if not self.class_filter:
            return True
        label = self.get_label(class_id)
        return label.lower() in self.class_filter

    def scale_box(self, box, original_shape):
        h, w = original_shape
        x, y, width, height = box
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        width = max(0, min(int(width), w - x))
        height = max(0, min(int(height), h - y))
        return x, y, width, height

