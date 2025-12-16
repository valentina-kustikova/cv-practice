from abc import ABC, abstractmethod


class ObjectDetector(ABC):
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.output_layers = []
        self.classes = []

    @abstractmethod
    def preprocess(self, image):
        """Подготовка изображения для конкретной сети"""
        pass

    @abstractmethod
    def postprocess(self, image, outs):
        """Обработка сырого выхода сети в список bounding box'ов"""
        pass

    def detect(self, image):
        """Основной метод запуска"""
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return self.postprocess(image, outs)
