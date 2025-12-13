import cv2
import numpy as np
from .base_detector import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self, model_path, config_path, classes_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.nms_threshold = nms_threshold
        super().__init__(model_path, config_path, classes_path, confidence_threshold)
    
    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), 
            swapRB=True, crop=False
        )
        return blob
    
    def postprocess(self, outputs, image_shape):
        height, width = image_shape[:2]
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': [x, y, x + w, y + h]
                })
        
        return results
    
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]