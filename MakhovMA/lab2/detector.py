import cv2
import numpy as np
import os
from models.yolo_detector import YOLODetector
from models.ssd_detector import SSDDetector


class VehicleDetector:
    """Основной класс для детектирования транспортных средств"""

    def __init__(self, model_type='yolo', model_path=None, config_path=None, classes_file=None):
        self.model_type = model_type.lower()

        if self.model_type == 'yolo':
            self.detector = YOLODetector(model_path, config_path, classes_file)
        elif self.model_type == 'ssd':
            self.detector = SSDDetector(model_path, classes_file)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def set_confidence_threshold(self, threshold):
        """Установка порога уверенности"""
        self.detector.confidence_threshold = threshold

    def detect_vehicles(self, image):
        """Детектирование транспортных средств"""
        return self.detector.detect(image)

    def calculate_metrics(self, detections, ground_truth, iou_threshold=0.5):
        """Вычисление метрик TPR и FDR с учетом классов"""
        true_positives = 0
        false_positives = 0
        total_ground_truth = len(ground_truth)

        matched_gt = set()

        for det in detections:
            det_bbox = det['bbox']
            matched = False

            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue

                gt_bbox = gt['bbox']
                iou = self._calculate_iou(det_bbox, gt_bbox)
                if iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(i)
                    matched = True
                    break

            if not matched:
                false_positives += 1

        false_negatives = total_ground_truth - len(matched_gt)

        tpr = true_positives / total_ground_truth if total_ground_truth > 0 else 0
        fdr = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        return tpr, fdr

    def _calculate_iou(self, bbox1, bbox2):
        """Вычисление Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Вычисление координат пересечения
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
        return iou