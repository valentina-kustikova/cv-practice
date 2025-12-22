from .base import Detector
import cv2
import numpy as np
import os

class YOLOv8Detector(Detector):
    """
    Детектор на основе YOLOv8 (ONNX формат).
    """
    
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov8l.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.input_size = (640, 640)
        self.conf_th = 0.5
        self.nms_th = 0.45
        
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        self.vehicle_classes = {'car', 'bus', 'truck', 'motorcycle'}

    def detect(self, image):
        if image is None: return []
        h, w = image.shape[:2]
        
        # Letterbox logic
        target_w, target_h = self.input_size
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (nw, nh))
        blob_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        pad_x, pad_y = (target_w - nw) // 2, (target_h - nh) // 2
        blob_img[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
        
        blob = cv2.dnn.blobFromImage(blob_img, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # [1, 84, 8400] -> [8400, 84]
        out = outputs[0].T 
        
        boxes = []
        scores = []
        class_ids = []
        
        # Ускоренная фильтрация
        # out[:, 4:] - вероятности классов
        classes_scores = out[:, 4:]
        max_scores = np.max(classes_scores, axis=1)
        argmax_indices = np.argmax(classes_scores, axis=1)
        
        mask = max_scores > self.conf_th
        out_filtered = out[mask]
        scores_filtered = max_scores[mask]
        classes_filtered = argmax_indices[mask]
        
        for i in range(len(out_filtered)):
            class_id = classes_filtered[i]
            if self.class_names[class_id] in self.vehicle_classes:
                row = out_filtered[i]
                cx, cy, bw, bh = row[:4]
                
                # Возврат к координатам оригинала
                x = (cx - bw/2 - pad_x) / scale
                y = (cy - bh/2 - pad_y) / scale
                bw /= scale
                bh /= scale
                
                boxes.append([int(x), int(y), int(bw), int(bh)])
                scores.append(float(scores_filtered[i]))
                class_ids.append(class_id)
        
        results = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_th, self.nms_th)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    results.append({
                        'class': self.class_names[class_ids[i]],
                        'confidence': scores[i],
                        'bbox': [x, y, x+w, y+h]
                    })
        return results