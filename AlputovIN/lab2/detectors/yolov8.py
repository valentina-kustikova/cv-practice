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
        
        # Загружаем модель с оптимизацией для CPU
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.input_size = (640, 640)
        self.conf_th = 0.5
        self.nms_th = 0.4
        # Кэш для letterbox (избегаем пересоздания массива)
        self._letterbox_cache = None
        # COCO classes (80)
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
        """
        Детектирует транспортные средства на изображении.
        
        Оптимизации:
        - Векторизованная фильтрация по confidence
        - Ранний выход при отсутствии детекций
        - Оптимизированная обработка координат
        """
        # Валидация входного изображения
        if image is None or image.size == 0:
            return []
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []
        
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []
        
        target_h, target_w = self.input_size
        
        # Letterbox resize с сохранением пропорций (оптимизировано)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Проверка на слишком маленькое изображение
        if new_w < 1 or new_h < 1:
            return []
        
        # Масштабируем изображение
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Создаём letterbox (переиспользуем кэш если возможно)
        letterbox = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Вычисляем отступы
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Помещаем изображение в центр
        letterbox[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Предобработка: BGR -> RGB, нормализация [0, 1]
        blob = cv2.dnn.blobFromImage(
            letterbox,
            scalefactor=1/255.0,
            size=self.input_size,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        # Инференс модели
        self.net.setInput(blob)
        outputs = self.net.forward()
        
        # Валидация вывода модели
        if not outputs or len(outputs) == 0:
            return []
        
        # YOLOv8 output shape: [1, 84, 8400] -> транспонируем в [8400, 84]
        output = outputs[0]
        if len(output.shape) == 3:
            output = output[0]
        output = output.T  # [8400, 84]
        
        # Векторизованная фильтрация по confidence (оптимизация)
        # Извлекаем все scores и находим максимальные confidence
        all_scores = output[:, 4:]  # [8400, 80]
        max_confidences = np.max(all_scores, axis=1)  # [8400]
        max_class_ids = np.argmax(all_scores, axis=1)  # [8400]
        
        # Фильтруем по confidence
        valid_mask = max_confidences >= self.conf_th
        if not np.any(valid_mask):
            return []
        
        # Применяем маску
        valid_outputs = output[valid_mask]
        valid_confidences = max_confidences[valid_mask]
        valid_class_ids = max_class_ids[valid_mask]
        
        # Фильтруем по классам транспорта (векторизованно)
        valid_class_names = np.array([self.class_names[cid] if cid < len(self.class_names) 
                                      else str(cid) for cid in valid_class_ids])
        vehicle_mask = np.array([name in self.vehicle_classes for name in valid_class_names])
        
        if not np.any(vehicle_mask):
            return []
        
        # Применяем маску транспорта
        vehicle_outputs = valid_outputs[vehicle_mask]
        vehicle_confidences = valid_confidences[vehicle_mask]
        vehicle_class_ids = valid_class_ids[vehicle_mask]
        
        # Векторизованное преобразование координат (оптимизация)
        bboxes = vehicle_outputs[:, :4]  # [N, 4] - cx, cy, w, h
        cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        
        # Преобразуем из центра в углы
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        
        # Убираем padding и масштабируем обратно
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        # Векторизованный клиппинг
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        # Фильтруем валидные bbox (x2 > x1, y2 > y1)
        valid_bbox_mask = (x2 > x1) & (y2 > y1)
        if not np.any(valid_bbox_mask):
            return []
        
        # Применяем маску
        x1 = x1[valid_bbox_mask]
        y1 = y1[valid_bbox_mask]
        x2 = x2[valid_bbox_mask]
        y2 = y2[valid_bbox_mask]
        vehicle_confidences = vehicle_confidences[valid_bbox_mask]
        vehicle_class_ids = vehicle_class_ids[valid_bbox_mask]
        
        # Подготавливаем для NMS
        boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] 
                 for i in range(len(x1))]
        confs = vehicle_confidences.tolist()
        class_ids = vehicle_class_ids.tolist()
        
        # Применяем NMS
        results = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_th, self.nms_th)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w_box, h_box = boxes[i]
                    # Дополнительная проверка размера bbox (фильтруем слишком маленькие)
                    if w_box > 2 and h_box > 2:  # Минимальный размер 2x2 пикселя
                        results.append({
                            'class': self.class_names[class_ids[i]],
                            'confidence': float(confs[i]),
                            'bbox': [x, y, x + w_box, y + h_box]
                        })
        
        return results
