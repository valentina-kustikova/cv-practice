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
        
        # Загружаем модель с оптимизацией для CPU
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.input_size = (416, 416)
        self.width = 416
        self.height = 416
        self.conf_th = 0.5
        self.nms_th = 0.4
        self.strides = [8, 16, 32, 64]
        self.num_classes = 80
        self.grid = np.arange(8, dtype=np.float32)  # Предвычисляем для оптимизации
        
        # Кэш для сеток каждого уровня (оптимизация)
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
        """
        Декодирует координаты из формата распределения (GFL - Generalized Focal Loss).
        
        """
        # Валидация входных данных
        if boxes_raw.size == 0 or len(boxes_raw.shape) != 2:
            return np.empty((0, 4), dtype=np.float32)
        
        boxes = []
        # Векторизованная обработка всех 4 координат
        for i in range(4):  # x1, y1, x2, y2
            # Извлекаем распределение для i-й координаты
            dist = boxes_raw[:, 8*i:8*(i+1)]  # [N, 8]
            # Применяем softmax (exp и нормализация)
            exp_dist = np.exp(dist - np.max(dist, axis=1, keepdims=True))  # Стабильность вычислений
            exp_sum = np.sum(exp_dist, axis=1, keepdims=True)
            # Вычисляем взвешенную сумму (expectation)
            wing = np.sum(exp_dist * self.grid, axis=1) / (exp_sum[:, 0] + 1e-8) * pitch
            
            # Вычисляем координаты в зависимости от типа
            if i == 0:
                coord = np.maximum(grid_x - wing, 0.0)  # x1
            elif i == 1:
                coord = np.maximum(grid_y - wing, 0.0)  # y1
            elif i == 2:
                coord = np.minimum(grid_x + wing, 1.0)  # x2
            else:
                coord = np.minimum(grid_y + wing, 1.0)  # y2
            
            boxes.append(coord.reshape(-1, 1))
        
        return np.concatenate(boxes, axis=1)

    def _get_grid_for_stride(self, stride):
        """Получает сетку для уровня, используя кэш для оптимизации."""
        if stride not in self._grid_cache:
            fh = self.height // stride
            fw = self.width // stride
            x = np.arange(fw, dtype=np.float32) + 0.5
            y = np.arange(fh, dtype=np.float32) + 0.5
            gx, gy = np.meshgrid(x, y)
            gx = gx.flatten() / fw
            gy = gy.flatten() / fh
            self._grid_cache[stride] = (gx, gy)
        return self._grid_cache[stride]

    def detect(self, image):
        """
        Детектирует транспортные средства на изображении.
        
        """
        # Валидация входного изображения
        if image is None or image.size == 0:
            return []
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []
        
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []
        
        # Предобработка изображения
        blob = cv2.dnn.blobFromImage(image, 1/57.375, self.input_size, (103.53,116.28,123.675), swapRB=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # Валидация вывода модели
        if not outputs or len(outputs) == 0 or outputs[0].size == 0:
            return []
        
        # Обрабатываем вывод модели по уровням (strides)
        outputs = outputs[0]
        start = 0
        boxes_list, scores_list = [], []
        
        for stride in self.strides:
            fh = self.height // stride
            fw = self.width // stride
            num_preds = fh * fw
            
            # Проверка границ массива
            if start + num_preds > len(outputs):
                break
            
            preds_level = outputs[start:start+num_preds]
            
            # Валидация структуры данных
            if preds_level.shape[1] < self.num_classes + 32:  # 80 классов + 32 для координат
                break
            
            scores_level = preds_level[:, :self.num_classes]
            boxes_level = preds_level[:, self.num_classes:]
            
            # Получаем сетку (с кэшированием)
            gx, gy = self._get_grid_for_stride(stride)
            pitch = stride / self.width
            
            # Декодируем координаты
            boxes_decoded = self.decode_boxes(boxes_level, gx, gy, pitch)
            
            # Валидация декодированных координат
            if boxes_decoded.size > 0:
                boxes_list.append(boxes_decoded)
                scores_list.append(scores_level)
            
            start += num_preds
        
        # Объединяем все уровни
        if not boxes_list:
            return []
        
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)
        
        # Векторизованное нахождение классов и confidence (оптимизация)
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        
        # Векторизованная фильтрация по confidence (оптимизация)
        conf_mask = confs >= self.conf_th
        if not np.any(conf_mask):
            return []
        
        # Применяем маску confidence
        boxes = boxes[conf_mask]
        confs = confs[conf_mask]
        class_ids = class_ids[conf_mask]
        
        # Фильтруем по классам транспорта (векторизованно)
        class_names_array = np.array([self.class_names[cid] if cid < len(self.class_names) 
                                     else str(cid) for cid in class_ids])
        vehicle_mask = np.array([name in self.vehicle_classes for name in class_names_array])
        
        if not np.any(vehicle_mask):
            return []
        
        # Применяем маску транспорта
        boxes = boxes[vehicle_mask]
        confs = confs[vehicle_mask]
        class_ids = class_ids[vehicle_mask]
        
        # Векторизованное преобразование координат в пиксели (оптимизация)
        x1_px = (boxes[:, 0] * w).astype(np.int32)
        y1_px = (boxes[:, 1] * h).astype(np.int32)
        x2_px = (boxes[:, 2] * w).astype(np.int32)
        y2_px = (boxes[:, 3] * h).astype(np.int32)
        
        # Векторизованная проверка валидности bbox
        valid_mask = (x2_px > x1_px) & (y2_px > y1_px) & (x1_px >= 0) & (y1_px >= 0) & \
                     (x2_px <= w) & (y2_px <= h) & \
                     (x2_px - x1_px > 2) & (y2_px - y1_px > 2)  # Минимальный размер
        
        if not np.any(valid_mask):
            return []
        
        # Применяем маску
        x1_px = x1_px[valid_mask]
        y1_px = y1_px[valid_mask]
        x2_px = x2_px[valid_mask]
        y2_px = y2_px[valid_mask]
        confs = confs[valid_mask]
        class_ids = class_ids[valid_mask]
        
        # Подготавливаем для NMS
        all_boxes = [[x1_px[i], y1_px[i], x2_px[i], y2_px[i]] for i in range(len(x1_px))]
        all_scores = confs.tolist()
        all_class_ids = class_ids.tolist()
        
        # Применяем NMS
        results = []
        if len(all_boxes) > 0:
            boxes_nms = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes]
            indices = cv2.dnn.NMSBoxes(boxes_nms, all_scores, self.conf_th, self.nms_th)
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = all_boxes[i]
                    class_id = int(all_class_ids[i])
                    results.append({
                        'class': self.class_names[class_id],
                        'confidence': float(all_scores[i]),
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return results
