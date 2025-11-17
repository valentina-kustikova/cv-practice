from typing import Dict, List
from detector_base import ModelConfig
from detector_base import BaseDetector
from yolo_detector import YOLODetector
from ssd_detector import SSDMobileNetDetector
from rcnn_detector import FasterRCNNDetector


class VehicleDetectorFactory:
    _MODELS: Dict[str, Dict] = {
        "yolov4tiny": {
            "name": "yolov4-tiny",
            "description": "yolov4-tiny trained on COCO dataset",
            "vehicle_classes": ["car", "truck", "bus", "motorcycle"],
            "model_path": "models/yolov4-tiny.weights",
            "config_path": "models/yolov4tiny.cfg",
            "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
            # Пример, должен соответствовать модели
            "input_size": (416, 416),
            "scale_factor": 1 / 255.0,
            "swap_rb": True,
            "nms_threshold": 0.4,
            "detector_class": YOLODetector,
        },

        "yolov4": {
            "name": "yolov4",
            "description": "yolov4 trained on COCO dataset",
            "vehicle_classes": ["car", "truck", "bus", "motorcycle"],
            "model_path": "models/yolov4.weights",
            "config_path": "models/yolov4.cfg",
            "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
            # Пример, должен соответствовать модели
            "input_size": (416, 416),
            "scale_factor": 1 / 255.0,
            "swap_rb": True,
            "nms_threshold": 0.4,
            "detector_class": YOLODetector,
        },

        "mobilenet": {
            "name": "SSD MobileNet V3",
            "description": "SSD with MobileNet V3",
            "vehicle_classes": ["car", "truck", "bus"],
            "model_path": "models/frozen_inference_graph.pb",
            "config_path": "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
            "classes": ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"],
            # Пример, должен соответствовать модели
            "input_size": (600, 600),
            "scale_factor": 1.0,
            "swap_rb": True,
            "nms_threshold": 0.4,
            "detector_class": SSDMobileNetDetector,
        },

        "rcnn": {
            "name": "Faster R-CNN",
            "description": "Faster R-CNN with Inception V2",
            "vehicle_classes": ["car", "truck", "bus"],
            "model_path": "models/rcnn_frozen_inference_graph.pb",
            "config_path": "models/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt",
            "classes": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
            # Пример, должен соответствовать модели
            "input_size": (300, 300),
            "scale_factor": 1.0,
            "swap_rb": True,
            "nms_threshold": 0.4,
            "detector_class": FasterRCNNDetector,
        },
    }

    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, str | list[str]]]:
        return {
            key: {
                "name": value["name"],
                "description": value["description"],
                "vehicle_classes": value["vehicle_classes"],
            }
            for key, value in VehicleDetectorFactory._MODELS.items()
        }

    @staticmethod
    def create_detector(model_type: str, confidence_threshold: float = 0.5) -> BaseDetector:
        if model_type not in VehicleDetectorFactory._MODELS:
            raise ValueError(f"Unknown model type: {model_type}")

        # Копируем конфиг, чтобы извлечь detector_class, не меняя оригинал
        model_info = VehicleDetectorFactory._MODELS[model_type].copy()

        # Извлекаем класс детектора
        detector_class = model_info.pop("detector_class")

        # Добавляем динамический порог
        model_info["confidence_threshold"] = confidence_threshold

        # **ИСПРАВЛЕНИЕ:** Используем распаковку словаря для инициализации ModelConfig.
        # Это лаконично и автоматически передает ВСЕ нужные поля, включая
        # 'vehicle_classes', 'description' и т.д.
        config = ModelConfig(**model_info)

        # Создаем и возвращаем экземпляр детектора
        return detector_class(config)