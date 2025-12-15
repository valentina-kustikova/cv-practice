from .yolov8 import YOLOv8Detector
from .fasterrcnn import FasterRCNNDetector
from .nanodet import NanoDetDetector

def get_detector(name):
    if name == 'yolov8':
        return YOLOv8Detector()
    elif name == 'fasterrcnn':
        return FasterRCNNDetector()
    elif name == 'nanodet':
        return NanoDetDetector()
    else:
        raise ValueError(f'Unknown detector: {name}')
