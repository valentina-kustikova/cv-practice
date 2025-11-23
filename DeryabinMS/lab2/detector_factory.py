from typing import Dict, Any, List
import os

from detector_base import DetectorConfig
from yolo_detector import YOLOv4Detector
from ssd_mobilenet_detector import SSDMobileNetDetector
from faster_rcnn_detector import FasterRCNNDetector


class VehicleDetectorFactory:
    """Фабрика для создания детекторов"""

    @staticmethod
    def _load_classes(names_path: str) -> List[str]:
        with open(names_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, Any]]:
        """Справочная информация"""
        return {
            "yolo": {
                "name": "YOLOv4",
                "vehicle_classes": ["car", "bus", "truck"]
            },
            "ssd": {
                "name": "SSD MobileNet v3",
                "vehicle_classes": ["car", "bus", "truck"]
            },
            "rcnn": {
                "name": "Faster R-CNN",
                "vehicle_classes": ["car", "bus", "truck"]
            },
        }

    @staticmethod
    def create_detector(model_key: str, confidence_threshold: float):
        """Создание детектора по ключу"""
        coco_names_path = os.path.join("configs", "coco.names")
        classes = VehicleDetectorFactory._load_classes(coco_names_path)

        if model_key == "yolo":
            config = DetectorConfig(
                name="YOLOv4",
                model_path=os.path.join("models", "yolov4.weights"),
                config_path=os.path.join("configs", "yolov4.cfg"),
                classes=classes,
                input_size=(416, 416),
                scale=1.0 / 255.0,
                swap_rb=True,
                mean=(0.0, 0.0, 0.0),
                conf_threshold=confidence_threshold,
                nms_threshold=0.4,
            )
            return YOLOv4Detector(config)

        if model_key == "ssd":
            config = DetectorConfig(
                name="SSD MobileNet v3",
                model_path=os.path.join("models", "frozen_inference_graph.pb"),
                config_path=os.path.join("configs", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"),
                classes=classes,
                input_size=(320, 320),
                scale=1.0,
                swap_rb=True,
                mean=(0.0, 0.0, 0.0),
                conf_threshold=confidence_threshold,
                nms_threshold=0.4,
            )
            return SSDMobileNetDetector(config)

        if model_key == "rcnn":
            config = DetectorConfig(
                name="Faster R-CNN",
                model_path=os.path.join("models", "frozen_inference_graph_faster_rcnn.pb"),
                config_path=os.path.join("configs", "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"),
                classes=classes,
                input_size=(300, 300),
                scale=1.0,
                swap_rb=True,
                mean=(0.0, 0.0, 0.0),
                conf_threshold=confidence_threshold,
                nms_threshold=0.4,
            )
            return FasterRCNNDetector(config)

        raise ValueError(f"Неизвестная модель: {model_key}")
