# src/detector_base.py

import cv2
import numpy as np
from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    def __init__(self, model_path, config_path, classes_path=None, conf_threshold=0.5, nms_threshold=0.4):
        self.model_path = model_path
        self.config_path = config_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = None
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        self.net = None

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, outputs, original_image_shape):
        pass

    def detect(self, image):
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        return self.postprocess(outs, image.shape)

    @abstractmethod
    def get_output_layers(self):
        pass
