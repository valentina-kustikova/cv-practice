from abc import ABC, abstractmethod
import cv2
import numpy as np

class ObjectDetector(ABC):
    """
    Базовый абстрактный класс для всех детекторов.
    Определяет общий интерфейс: загрузка, детектирование, отрисовка.
    """
    def __init__(self, model_path, config_path, classes_path, conf_thresh=0.5, nms_thresh=0.4):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.classes = self._load_class_names(classes_path)
        self.net = self._load_network(model_path, config_path)
        
        # Генерируем цвета для классов один раз при инициализации
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

    def _load_class_names(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    @abstractmethod
    def _load_network(self, model, config):
        pass

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, image, outputs):
        """Должен возвращать список словарей: {'class': str, 'conf': float, 'box': (x1, y1, x2, y2)}"""
        pass

    def detect(self, image):
        blob = self.preprocess(image)
        self.net.setInput(blob)
        
        # Получаем имена выходных слоев
        out_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(out_names)
        
        return self.postprocess(image, outputs)

    def draw_results(self, image, detections):
        img_cp = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class']} {det['conf']:.3f}"
            
            # Получаем ID класса для цвета (если есть в списке, иначе белый)
            try:
                class_id = self.classes.index(det['class'])
                color = [int(c) for c in self.colors[class_id]]
            except ValueError:
                color = (255, 255, 255)

            cv2.rectangle(img_cp, (x1, y1), (x2, y2), color, 2)
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cp, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_cp, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return img_cp