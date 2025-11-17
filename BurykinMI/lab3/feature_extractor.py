import cv2
import numpy as np

from abstract import FeatureExtractor


# ============================================================================
# Реализует конкретный экстрактор признаков на основе OpenCV.
# Класс OpenCVFeatureExtractor поддерживает детекторы SIFT, ORB и AKAZE.
# Извлекает ключевые точки и дескрипторы из изображений в градациях серого.
# Используется в стратегии Bag of Words для построения визуального словаря.
# ============================================================================

class OpenCVFeatureExtractor(FeatureExtractor):
    """Экстрактор признаков на основе детекторов OpenCV"""

    def __init__(self, detector_type: str = 'sift'):
        """
        Args:
            detector_type: Тип детектора ('sift', 'orb', 'akaze')
        """
        self.detector_type = detector_type
        self.detector = self._create_detector()

    def _create_detector(self):
        """Создать детектор OpenCV"""
        if self.detector_type == 'sift':
            return cv2.SIFT_create()
        elif self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'akaze':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Неизвестный детектор: {self.detector_type}")

    def extract(self, image_path: str) -> np.ndarray:
        """Извлечь дескрипторы из изображения"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return descriptors

    def get_name(self) -> str:
        return self.detector_type
