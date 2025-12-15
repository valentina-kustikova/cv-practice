from typing import List, Tuple
import cv2
from detector_base import Detection

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

        cv2.putText(result, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Черный текст

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