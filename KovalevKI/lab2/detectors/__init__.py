from .base import ObjectDetector
from .ssd import MobileNetSSDDetector
from .yolo_v5_onnx import YOLOv5sOpenCVDetector

DETECTORS = {
    "mobilenet_ssd": MobileNetSSDDetector,
    "yolov5s_onnx": YOLOv5sOpenCVDetector,
}