# detectors.py
import cv2
import numpy as np
import os

class BaseDetector:
    def __init__(self, model_path, input_size):
        self.model_path = model_path
        self.input_size = input_size
        self.net = None


    def preprocess(self, img):
        raise NotImplementedError

    def forward(self, blob):
        self.net.setInput(blob)
        return self.net.forward()

    def postprocess(self, outputs, img):
        raise NotImplementedError

    def detect(self, img):
        blob = self.preprocess(img)
        outputs = self.forward(blob)
        return self.postprocess(outputs, img)
