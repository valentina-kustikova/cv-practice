import cv2
import numpy as np

from abstract import FeatureExtractor


class OpenCVFeatureExtractor(FeatureExtractor):
    """Экстрактор признаков на основе детекторов OpenCV"""

    def __init__(self, detector_type: str = 'sift'):
        self.detector_type = detector_type
        self.detector = self._create_detector()

    def _create_detector(self):
        """Создать детектор OpenCV (Фабричный метод)"""
        if self.detector_type == 'sift':
            # SIFT: Инвариантен к масштабу, использует градиенты. Вектор 128 float.
            return cv2.SIFT_create()
        elif self.detector_type == 'orb':
            # ORB: Быстрый, бинарный дескриптор. Менее точен для текстур.
            return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'akaze':
            # AKAZE: Нелинейное пространство масштабов.
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Неизвестный детектор: {self.detector_type}")

    def extract(self, image_path: str) -> np.ndarray:
        """Извлечь дескрипторы из изображения"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None

        # Переводим в оттенки серого.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detectAndCompute возвращает:
        # keypoints: координаты (x, y) точек (здесь не нужны)
        # descriptors: матрицы признаков для каждой точки
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return descriptors

    def get_name(self) -> str:
        return self.detector_type
