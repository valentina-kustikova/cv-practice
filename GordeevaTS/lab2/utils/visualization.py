import cv2
import numpy as np
from typing import List

def draw_detections(image: np.ndarray, detections: List[list], 
                   vehicle_classes: List[str] = None):
    if vehicle_classes is None:
        vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
    
    result_image = image.copy()
    
    for detection in detections:
        if len(detection) < 7:
            continue
            
        x1, y1, x2, y2, class_id, confidence, class_name = detection
        
        if class_name not in vehicle_classes:
            continue
        color = _get_color_for_class(class_name)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {confidence:.3f}"

        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.rectangle(result_image, 
                     (x1, y1 - label_height - baseline - 5),
                     (x1 + label_width, y1),
                     color, -1)

        cv2.putText(result_image, label, (x1, y1 - baseline - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_image

def _get_color_for_class(class_name: str):
    color_map = {
        'car': (0, 255, 0),
        'bus': (255, 0, 0),
        'truck': (0, 0, 255),
        'motorcycle': (255, 255, 0), 
        'bicycle': (255, 0, 255)   
    }
    return color_map.get(class_name, (128, 128, 128)) 