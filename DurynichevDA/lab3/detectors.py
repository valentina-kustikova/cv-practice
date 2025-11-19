import cv2

class SIFTDetector:
    def __init__(self):
        self.detector = cv2.SIFT_create(
            nfeatures=800, contrastThreshold=0.04, edgeThreshold=10
        )
        self.color = (0, 0, 255)      # ярко-красный
        self.name = "SIFT"

    def detect_and_compute(self, gray):
        return self.detector.detectAndCompute(gray, None)


class ORBDetector:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=600)
        self.color = (0, 165, 255)    # оранжевый
        self.name = "ORB"

    def detect_and_compute(self, gray):
        kp = self.detector.detect(gray, None)
        kp, des = self.detector.compute(gray, kp)
        return kp, des


class AKAZEDetector:
    def __init__(self):
        self.detector = cv2.AKAZE_create(threshold=0.0015)
        self.color = (0, 255, 0)      # зелёный
        self.name = "AKAZE"

    def detect_and_compute(self, gray):
        return self.detector.detectAndCompute(gray, None)