import os
from typing import List, Tuple
import cv2
from detector_base import Detection

class AnnotationLoader:
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> dict[str, List[Tuple[int, int, int, int]]]:
        annotations: dict[str, List[Tuple[int, int, int, int]]] = {}
        if not os.path.exists(self.annotation_path):
            return annotations

        with open(self.annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[1].lower() == "car":
                    image_id = parts[0].zfill(6)
                    x1, y1, x2, y2 = map(int, parts[2:6])
                    if image_id not in annotations:
                        annotations[image_id] = []
                    annotations[image_id].append((x1, y1, x2, y2))

        return annotations

    def get_ground_truth(self, image_id: str) -> List[Tuple[int, int, int, int]]:
        return self.annotations.get(image_id, [])

class DetectionEvaluator:

    def __init__(self, target_class: str = "car"):
        self.target_class = target_class.lower()
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def evaluate_frame(
        self,
        detections: List[Detection],
        ground_truth: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.5
    ) -> Tuple[float, float]:
        frame_tp = 0
        frame_fp = 0
        gt_used = [False] * len(ground_truth)

        detections = sorted(
            [d for d in detections if d.class_name.lower() == self.target_class],
            key=lambda x: x.confidence,
            reverse=True
        )

        for detection in detections:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(ground_truth):
                if not gt_used[i]:
                    iou = self.calculate_iou(detection.bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx != -1:
                frame_tp += 1
                self.true_positives += 1
                gt_used[best_gt_idx] = True
            else:
                frame_fp += 1
                self.false_positives += 1

        frame_fn = sum(1 for used in gt_used if not used)
        self.false_negatives += frame_fn

        frame_tpr = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
        frame_fdr = frame_fp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0

        return frame_tpr, frame_fdr

    def get_metrics(self) -> Tuple[float, float]:
        tpr = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        fdr = self.false_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        return tpr, fdr

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0


CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 0, 255),
    "bus": (255, 0, 0),
    "motorcycle": (0, 255, 255),
    "person": (255, 0, 255),
    "bicycle": (255, 255, 0),
}
DEFAULT_COLOR = (255, 255, 255)

def draw_detections(image, detections: List[Detection]):
    result = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        class_name_lower = detection.class_name.lower()
        label = f"{class_name_lower}: {detection.confidence:.3f}"

        color = CLASS_COLORS.get(class_name_lower, DEFAULT_COLOR)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)

        cv2.putText(result, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return result

def draw_ground_truth(image, ground_truth: List[Tuple[int, int, int, int]], color: Tuple[int, int, int] = (255, 0, 255)):

    result = image.copy()
    for bbox in ground_truth:
        x1, y1, x2, y2 = bbox
        label = "GT"

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x2 - text_width
        text_y = y1 + text_height + baseline + 5
        cv2.rectangle(result, (text_x, text_y - text_height - baseline - 5), (text_x + text_width, text_y), color, -1)
        cv2.putText(result, label, (text_x, text_y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result

def _draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def _put_text_with_shadow(img, text, org, font, font_scale, color, thickness, shadow_offset=(1, 1), shadow_color=(0, 0, 0)):
    x, y = org
    dx, dy = shadow_offset
    cv2.putText(img, text, (x + dx, y + dy), font, font_scale, shadow_color, thickness + 1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

def display_metrics(
    image,
    tpr: float,
    fdr: float,
    frame_tpr: float | None = None,
    frame_fdr: float | None = None,
    position: str = "bottom-left",  # или "top-left", "bottom-right", "bottom-left"
    margin: int = 20,
    bg_alpha: float = 0.6,
):
    result = image.copy()
    h, w = image.shape[:2]

    lines = ["Metrics:"]
    lines.append(f"TPR: {tpr:.3f}")
    lines.append(f"FDR: {fdr:.3f}")
    if frame_tpr is not None and frame_fdr is not None:
        lines.append(f"Frame TPR: {frame_tpr:.3f}")
        lines.append(f"Frame FDR: {frame_fdr:.3f}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thickness = 1
    line_height = 28
    text_width = max(cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines)
    box_width = text_width + 30
    box_height = len(lines) * line_height + 15

    if position == "top-right":
        x1, y1 = w - box_width - margin, margin
    elif position == "top-left":
        x1, y1 = margin, margin
    elif position == "bottom-right":
        x1, y1 = w - box_width - margin, h - box_height - margin
    elif position == "bottom-left":
        x1, y1 = margin, h - box_height - margin
    else:
        raise ValueError("position must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
    x2, y2 = x1 + box_width, y1 + box_height

    overlay = result.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 30), -1)
    cv2.addWeighted(overlay, bg_alpha, result, 1 - bg_alpha, 0, result)

    _draw_rounded_rectangle(result, (x1, y1), (x2, y2), (80, 160, 255), 1, radius=8)

    for i, line in enumerate(lines):
        if i == 0:
            _put_text_with_shadow(
                result, line,
                (x1 + 15, y1 + 25 + i * line_height),
                font, font_scale + 0.1, (220, 240, 255), 2
            )
        else:
            _put_text_with_shadow(
                result, line,
                (x1 + 15, y1 + 25 + i * line_height),
                font, font_scale, (240, 255, 240), font_thickness
            )

    return result
