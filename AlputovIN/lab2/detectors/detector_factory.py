from .yolov8 import YOLOv8Detector
from .yolov4 import YOLOv4Detector
from .nanodet import NanoDetDetector

def get_detector(name):
    if name == 'yolov8':
        return YOLOv8Detector()
    elif name == 'yolov4':
        return YOLOv4Detector()
    elif name == 'nanodet':
        return NanoDetDetector()
    else:
        raise ValueError(f'Unknown detector: {name}')