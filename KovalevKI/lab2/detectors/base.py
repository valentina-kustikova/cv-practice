from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    def __init__(self, model_path, config_path=None, classes_path=None, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = []
        if classes_path:
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        self.model = self._load_model(model_path, config_path)

    @abstractmethod
    def _load_model(self, model_path, config_path):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, outputs, image_shape, *extras):
        pass

    @abstractmethod
    def detect(self, image):
        pass

    def get_class_name(self, class_id):
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"class_{class_id}"