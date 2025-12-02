import cv2
from abc import ABC, abstractmethod


class AbstractDetector(ABC):
    @abstractmethod
    def detect_and_compute(self, gray_image):
        """Возвращает (keypoints, descriptors)"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def color(self) -> tuple:
        """BGR цвет для визуализации"""
        pass


class SIFTDetector(AbstractDetector):
    def __init__(self):
        self.detector = cv2.SIFT_create(
            nfeatures=800, contrastThreshold=0.04, edgeThreshold=10
        )

    def detect_and_compute(self, gray_image):
        return self.detector.detectAndCompute(gray_image, None)

    @property
    def name(self):
        return "SIFT"

    @property
    def color(self):
        return (0, 0, 255)  # ярко-красный


class ORBDetector(AbstractDetector):
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=600)

    def detect_and_compute(self, gray_image):
        kp = self.detector.detect(gray_image, None)
        kp, des = self.detector.compute(gray_image, kp)
        return kp, des

    @property
    def name(self):
        return "ORB"

    @property
    def color(self):
        return (0, 165, 255)  # оранжевый

    def detect_and_compute(self, gray_image):
        kp = self.detector.detect(gray_image, None)
        kp, des = self.detector.compute(gray_image, kp)
        return kp, des


class AKAZEDetector(AbstractDetector):
    def __init__(self):
        self.detector = cv2.AKAZE_create(threshold=0.0015)

    def detect_and_compute(self, gray_image):
        return self.detector.detectAndCompute(gray_image, None)

    @property
    def name(self):
        return "AKAZE"

    @property
    def color(self):
        return (0, 255, 0)  # зелёный