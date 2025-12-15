from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def detect(self, image):
        """
        Вход: изображение (numpy array)
        Выход: список словарей [{'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}]
        """
        pass