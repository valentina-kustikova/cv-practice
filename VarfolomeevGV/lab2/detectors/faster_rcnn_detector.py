"""
Faster R-CNN детектор объектов.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

from .base_detector import BaseDetector


class FasterRCNNDetector(BaseDetector):
    """Детектор объектов на основе Faster R-CNN."""
    
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Инициализация Faster R-CNN детектора.
        
        Args:
            conf_threshold: Порог уверенности
            nms_threshold: Порог IoU для NMS (обычно не используется, т.к. NMS уже применен в модели)
        """
        super().__init__(conf_threshold, nms_threshold)
        # Faster R-CNN обычно работает с изображениями разных размеров
        # Стандартный размер для Inception v2: 800x600 или сохраняется пропорция
        self.input_size = (800, 600)
        # ImageNet mean для нормализации (BGR)
        self.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    
    def load_model(self, weights_path: str, config_path: Optional[str] = None,
                   names_path: Optional[str] = None) -> None:
        """
        Загрузка Faster R-CNN модели.
        
        Args:
            weights_path: Путь к .pb файлу (frozen graph)
            config_path: Путь к .pbtxt файлу (текстовое описание для OpenCV)
            names_path: Путь к файлу с названиями классов
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл модели не найден: {weights_path}")
        
        if config_path is None:
            raise ValueError("Для Faster R-CNN требуется config_path (.pbtxt файл)")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        # Загрузка TensorFlow модели
        self.net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
        
        # Установка backend
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Загрузка названий классов
        if names_path and os.path.exists(names_path):
            with open(names_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # Стандартные классы COCO
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        
        self.model_loaded = True
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Предобработка изображения для Faster R-CNN.
        
        Алгоритм:
        1. Изменение размера с сохранением пропорций (до 800x600 или максимальная сторона 800)
        2. Нормализация ImageNet: вычитание среднего значения (BGR: [103.939, 116.779, 123.68])
        3. Преобразование в формат NCHW
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            blob: Подготовленный blob
            metadata: Словарь с scale для постобработки
        """
        h, w = image.shape[:2]
        
        # Faster R-CNN обычно сохраняет пропорции
        # Стандартный подход: максимальная сторона = 800
        max_size = 800
        scale = min(max_size / h, max_size / w, 1.0)  # Не увеличиваем маленькие изображения
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize с сохранением пропорций
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Нормализация ImageNet (вычитание среднего)
        blob = resized.astype(np.float32)
        blob = blob - self.mean
        
        # Преобразование в формат NCHW (batch, channels, height, width)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        
        # Метаданные для постобработки
        metadata = {
            'scale': scale,
            'original_size': (w, h),
            'resized_size': (new_w, new_h)
        }
        
        return blob, metadata
    
    def postprocess(self, outputs, metadata: dict,
               original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка выходов Faster R-CNN.
        """
        # Faster R-CNN обычно возвращает один выход
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        
        if output is None or output.size == 0:
            return []
        
        # Убираем batch dimension
        if len(output.shape) == 4:
            # Обычно [1, 1, num_detections, 7] для TF OD API
            output = output[0][0]
        elif len(output.shape) == 3:
            output = output[0]  # [num_detections, features]
        elif len(output.shape) == 2:
            output = output  # Уже [num_detections, features]
        else:
            print(f"WARNING: Неподдерживаемая размерность выхода: {output.shape}")
            return []
        
        # Проверяем количество признаков
        num_features = output.shape[1] if len(output.shape) > 1 else output.shape[0]
        
        if num_features == 7:
            # Формат TensorFlow/SSD: [batch_id, class_id, confidence, x_min, y_min, x_max, y_max]
            batch_ids = output[:, 0].astype(np.int32)
            class_ids = output[:, 1].astype(np.int32)
            confidences = output[:, 2]
            boxes = output[:, 3:7]  # [x1, y1, x2, y2] в нормализованных координатах
            
            # Удаляем детекции, относящиеся не к текущему изображению
            valid_mask = batch_ids == 0
            boxes = boxes[valid_mask]
            class_ids = class_ids[valid_mask]
            confidences = confidences[valid_mask]
        elif num_features == 6:
            # Формат: [class_id, confidence, x1, y1, x2, y2]
            class_ids = output[:, 0].astype(np.int32)
            confidences = output[:, 1]
            boxes = output[:, 2:6]  # [x1, y1, x2, y2]
        elif num_features == 4:
            # Только координаты - возможно отдельные выходы
            boxes = output[:, :4]
            # Нужно получить class_ids и confidences из других выходов
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
                class_ids = outputs[1].flatten().astype(np.int32)
                confidences = outputs[2].flatten()
            else:
                return []
        else:
            # Неизвестный формат - пробуем стандартный
            print(f"WARNING: Неизвестный формат выхода Faster R-CNN: shape={output.shape}")
            # Пробуем извлечь как [class_id, confidence, x1, y1, x2, y2]
            if num_features >= 6:
                class_ids = output[:, 0].astype(np.int32)
                confidences = output[:, 1]
                boxes = output[:, 2:6]
            else:
                return []
        
        # Фильтрация по confidence
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        if len(boxes) == 0:
            return []
        
        # Обработка координат (нормализованные vs пиксельные)
        # Большинство моделей TF OD API возвращают нормализованные координаты [0, 1]
        if np.max(boxes) <= 1.0:
            # Нормализованные -> пиксели исходного изображения
            boxes[:, 0] *= original_shape[1]  # x1 * width
            boxes[:, 1] *= original_shape[0]  # y1 * height
            boxes[:, 2] *= original_shape[1]  # x2 * width
            boxes[:, 3] *= original_shape[0]  # y2 * height
        else:
            # Пиксели (относительно resized изображения) -> пиксели исходного
            scale = metadata['scale']
            if scale > 0:
                boxes = boxes / scale
        
        # Ограничение координат границами изображения
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_shape[0])  # y2
        
        # Формирование результата
        detections = []
        for i in range(len(boxes)):
            detections.append([
                float(boxes[i][0]),
                float(boxes[i][1]),
                float(boxes[i][2]),
                float(boxes[i][3]),
                int(class_ids[i]),
                float(confidences[i])
            ])
        
        return detections

