from .yolov4_tiny import YoloV4TinyDetector
from .mobilenet_ssd import MobilenetSsdDetector
from .ssd_inception import SsdInceptionDetector

MODEL_REGISTRY = {
    "yolov4_tiny": YoloV4TinyDetector,
    "mobilenet_ssd": MobilenetSsdDetector,
    "ssd_inception": SsdInceptionDetector,
}

