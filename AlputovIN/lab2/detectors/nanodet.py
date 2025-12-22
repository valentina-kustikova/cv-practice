from .base import Detector
import cv2
import numpy as np
import os

class NanoDetDetector(Detector):
    """
    Детектор на основе NanoDet-Plus (ONNX формат).
    """
    
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'nanodet-plus-m_416.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.input_size = (416, 416)
        self.width = 416
        self.height = 416
        self.conf_th = 0.45
        self.nms_th = 0.4
        self.strides = [8, 16, 32, 64]
        self.num_classes = 80
        self.grid = np.arange(8, dtype=np.float32)
        
        self._grid_cache = {}
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

    def decode_boxes(self, boxes_raw, grid_x, grid_y, pitch):
        if boxes_raw.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        
        boxes = []
        for i in range(4):
            dist = boxes_raw[:, 8*i:8*(i+1)]
            exp_dist = np.exp(dist - np.max(dist, axis=1, keepdims=True))
            exp_sum = np.sum(exp_dist, axis=1, keepdims=True)
            wing = np.sum(exp_dist * self.grid, axis=1) / (exp_sum[:, 0] + 1e-8) * pitch
            
            if i == 0: coord = grid_x - wing
            elif i == 1: coord = grid_y - wing
            elif i == 2: coord = grid_x + wing
            else: coord = grid_y + wing
            boxes.append(coord.reshape(-1, 1))
        
        # Клиппинг отрицательных значений в промежуточных вычислениях
        x1 = np.maximum(boxes[0], 0)
        y1 = np.maximum(boxes[1], 0)
        x2 = np.minimum(boxes[2], 1.0) # В нормализованных координатах не должно быть > 1, но пока держим relative
        y2 = np.minimum(boxes[3], 1.0)

        return np.concatenate([x1, y1, x2, y2], axis=1)

    def _get_grid_for_stride(self, stride):
        if stride not in self._grid_cache:
            fh = self.height // stride
            fw = self.width // stride
            x = np.arange(fw, dtype=np.float32) + 0.5
            y = np.arange(fh, dtype=np.float32) + 0.5
            gx, gy = np.meshgrid(x, y)
            self._grid_cache[stride] = (gx.flatten() / fw, gy.flatten() / fh)
        return self._grid_cache[stride]

    def detect(self, image):
        if image is None or image.size == 0: return []
        h, w = image.shape[:2]
        
        # Mean & Std для NanoDet
        blob = cv2.dnn.blobFromImage(image, 1/57.375, self.input_size, (103.53,116.28,123.675), swapRB=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        if not outputs or outputs[0].size == 0: return []
        
        outputs = outputs[0]
        start = 0
        boxes_list, scores_list = [], []
        
        for stride in self.strides:
            fh, fw = self.height // stride, self.width // stride
            num_preds = fh * fw
            if start + num_preds > len(outputs): break
            
            preds = outputs[start:start+num_preds]
            scores_lvl = preds[:, :self.num_classes]
            boxes_lvl = preds[:, self.num_classes:]
            
            gx, gy = self._get_grid_for_stride(stride)
            pitch = stride / self.width
            
            decoded = self.decode_boxes(boxes_lvl, gx, gy, pitch)
            boxes_list.append(decoded)
            scores_list.append(scores_lvl)
            start += num_preds
        
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        
        mask = confs >= self.conf_th
        boxes, confs, class_ids = boxes[mask], confs[mask], class_ids[mask]
        
        # Фильтрация по типам транспорта
        vehicle_indices = []
        for i, cid in enumerate(class_ids):
            if self.class_names[cid] in self.vehicle_classes:
                vehicle_indices.append(i)
        
        if not vehicle_indices: return []
        
        boxes = boxes[vehicle_indices]
        confs = confs[vehicle_indices]
        class_ids = class_ids[vehicle_indices]
        
        # В пиксели
        x1 = (boxes[:, 0] * w).astype(int)
        y1 = (boxes[:, 1] * h).astype(int)
        x2 = (boxes[:, 2] * w).astype(int)
        y2 = (boxes[:, 3] * h).astype(int)
        
        results = []
        # NMS
        final_boxes = []
        for i in range(len(x1)):
            final_boxes.append([x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]])
            
        indices = cv2.dnn.NMSBoxes(final_boxes, confs.tolist(), self.conf_th, self.nms_th)
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'class': self.class_names[class_ids[i]],
                    'confidence': float(confs[i]),
                    'bbox': [x1[i], y1[i], x2[i], y2[i]]
                })
        
        return results