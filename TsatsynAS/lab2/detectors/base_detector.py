import cv2
import numpy as np
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    def __init__(self, model_path, config_path=None, conf_threshold=0.5, nms_threshold=0.4):
        self.model_path = model_path
        self.config_path = config_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.classes = []
        self.colors = []

        self.load_model()
        self.generate_colors()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, outputs, image_shape):
        pass

    def generate_colors(self):
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, image):
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self.get_output_layers())

        print(f"Number of output layers: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} type: {type(output)}, shape: {output.shape}")

        return self.postprocess(outputs, image.shape)

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
