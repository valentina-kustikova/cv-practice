"""
Библиотека детекторов объектов для OpenCV DNN.
"""

from .base_detector import BaseDetector
from .yolo_detector import YOLODetector
from .faster_rcnn_detector import FasterRCNNDetector
from .retinanet_detector import RetinaNetDetector

__all__ = [
    'BaseDetector',
    'YOLODetector',
    'FasterRCNNDetector',
    'RetinaNetDetector',
]

