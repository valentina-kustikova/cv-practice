"""
YOLO детектор объектов.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

from .base_detector import BaseDetector
from utils.nms import nms_per_class


class YOLODetector(BaseDetector):
    """Детектор объектов на основе YOLOv5."""
    
    def __init__(self, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Инициализация YOLO детектора.
        
        Args:
            conf_threshold: Порог уверенности
            nms_threshold: Порог IoU для NMS
        """
        super().__init__(conf_threshold, nms_threshold)
        # YOLOv5 обычно работает с изображениями 640x640
        self.input_size = (640, 640)
    
    def load_model(self, weights_path: str, config_path: Optional[str] = None,
                   names_path: Optional[str] = None) -> None:
        """
        Загрузка YOLO модели.
        
        Args:
            weights_path: Путь к .onnx файлу
            config_path: Не используется для ONNX (оставлен для совместимости)
            names_path: Путь к файлу с названиями классов
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл модели не найден: {weights_path}")
        
        # Загрузка ONNX модели
        self.net = cv2.dnn.readNetFromONNX(weights_path)
        
        # Установка backend
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Загрузка названий классов
        if names_path and os.path.exists(names_path):
            with open(names_path, 'r', encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            # Стандартные классы COCO (80 классов)
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
        Предобработка изображения для YOLO.
        
        Алгоритм (letterbox):
        1. Изменение размера с сохранением пропорций до 640x640
        2. Добавление padding серым цветом (114, 114, 114) для получения квадрата
        3. Нормализация [0, 1] (деление на 255.0)
        4. Преобразование в формат NCHW
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            blob: Подготовленный blob
            metadata: Словарь с метаданными для постобработки (scale, pad)
        """
        h, w = image.shape[:2]
        target_size = self.input_size[0]  # 640
        
        # Вычисление масштаба с сохранением пропорций
        scale = min(target_size / h, target_size / w, 1.0)  # Не увеличиваем маленькие изображения
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize с сохранением пропорций
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Вычисление padding для получения квадрата 640x640
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        pad_h_remainder = target_size - new_h - pad_h
        pad_w_remainder = target_size - new_w - pad_w
        
        # Добавление padding серым цветом (114, 114, 114) - стандартный цвет для YOLO
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, pad_h_remainder,
            pad_w, pad_w_remainder,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        # Конвертация BGR → RGB (как в рабочем коде с swapRB=True)
        padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Нормализация [0, 1]
        blob = padded_rgb.astype(np.float32) / 255.0
        
        # Преобразование в формат NCHW (batch, channels, height, width)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        
        # Метаданные для постобработки
        metadata = {
            'scale': scale,
            'pad': (pad_w, pad_h),  # (pad_w, pad_h)
            'original_size': (w, h),
            'resized_size': (new_w, new_h),
            'target_size': target_size
        }
        
        return blob, metadata
    
    def postprocess(self, outputs, metadata: dict,
                   original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка выходов YOLO.
        
        YOLOv5 выход: [1, 25200, 85] где 85 = 4 координаты + 1 confidence + 80 классов
        Координаты в формате YOLO: [x_center, y_center, width, height] нормализованные [0, 1]
        
        Args:
            outputs: Выходные тензоры сети
            metadata: Метаданные из preprocess
            original_shape: Исходный размер изображения (height, width)
            
        Returns:
            Список детекций в формате [x1, y1, x2, y2, class_id, confidence]
        """
        # YOLOv5 обычно возвращает один выход
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]
        else:
            output = outputs
        
        if output is None or output.size == 0:
            return []
        
        # Убираем batch dimension и обрабатываем разные форматы
        if len(output.shape) == 3:
            # [1, 84, 8400] или [1, 85, N] - транспонируем в [N, features]
            if output.shape[1] == 84 or output.shape[1] == 85:
                # Формат [batch, features, num_detections] -> транспонируем
                output = output[0].transpose(1, 0)  # [num_detections, features]
            else:
                # Формат [batch, num_detections, features]
                output = output[0]
        elif len(output.shape) == 2:
            output = output
        else:
            print(f"WARNING: Неподдерживаемая размерность выхода YOLO: {output.shape}")
            return []
        
        # Проверяем формат выхода
        num_features = output.shape[1] if len(output.shape) > 1 else output.shape[0]
        
        if num_features == 85:
            # Стандартный формат: [x_center, y_center, width, height, confidence, class_probs...]
            # Координаты в абсолютных пикселях относительно сетки 640x640
            boxes_yolo = output[:, :4]  # [x_center, y_center, width, height] в пикселях
            confidences_raw = output[:, 4]  # raw confidence
            class_probs_raw = output[:, 5:]  # class probabilities
            
            # Проверяем, нужно ли применять sigmoid
            # Если значения уже в диапазоне [0, 1], sigmoid не нужен
            if np.max(confidences_raw) <= 1.0 and np.min(confidences_raw) >= 0.0:
                confidences = confidences_raw
            else:
                confidences = 1.0 / (1.0 + np.exp(-confidences_raw))
                
            if np.max(class_probs_raw) <= 1.0 and np.min(class_probs_raw) >= 0.0:
                class_probs = class_probs_raw
            else:
                class_probs = 1.0 / (1.0 + np.exp(-class_probs_raw))
                
        elif num_features == 84:
            # Формат без отдельного confidence: [x_center, y_center, width, height, class_probs...]
            # Координаты в абсолютных пикселях относительно сетки 640x640
            boxes_yolo = output[:, :4]  # [x_center, y_center, width, height] в пикселях
            class_probs_raw = output[:, 4:]  # class probabilities
            
            # Проверяем, нужно ли применять sigmoid
            # В данном случае значения очень маленькие (~1e-7), что говорит о том, что это вероятности
            # Применение sigmoid к 0 дает 0.5, что создает ложные срабатывания
            # Поэтому убираем принудительный sigmoid, если значения выглядят как вероятности
            if np.max(class_probs_raw) <= 1.0 and np.min(class_probs_raw) >= 0.0:
                class_probs = class_probs_raw
            else:
                class_probs = 1.0 / (1.0 + np.exp(-class_probs_raw))  # sigmoid
            
            # confidence = максимальная вероятность класса
            confidences = np.max(class_probs, axis=1)
        else:
            print(f"WARNING: Неожиданный формат выхода YOLO: shape={output.shape}")
            if num_features < 4:
                return []
            # Пробуем обработать как формат с 4 координатами
            boxes_yolo = output[:, :4]
            if num_features >= 5:
                confidences = output[:, 4]
                class_probs = output[:, 5:] if num_features > 5 else None
            else:
                confidences = np.ones(output.shape[0])
                class_probs = None
        
        # Фильтрация по confidence
        mask = confidences >= self.conf_threshold
        boxes_yolo = boxes_yolo[mask]
        confidences = confidences[mask]
        if class_probs is not None:
            class_probs = class_probs[mask]
        
        if len(boxes_yolo) == 0:
            return []
        
        # Получение класса с максимальной вероятностью
        if class_probs is not None and class_probs.shape[1] > 0:
            class_ids = np.argmax(class_probs, axis=1)
            max_class_probs = np.max(class_probs, axis=1)
            # Для формата 84: confidence уже = max_class_probs, не умножаем на себя
            # Для формата 85: умножаем max_class_probs на confidence
            if num_features == 84:
                class_scores = max_class_probs  # confidence уже вычислен как max_class_probs
            else:
                class_scores = max_class_probs * confidences  # Умножаем на confidence
        else:
            # Если нет вероятностей классов, используем confidence как score
            class_ids = np.zeros(len(boxes_yolo), dtype=np.int32)
            class_scores = confidences
        
        # Фильтрация по финальному score (confidence * class_prob)
        mask = class_scores >= self.conf_threshold
        boxes_yolo = boxes_yolo[mask]
        class_ids = class_ids[mask]
        class_scores = class_scores[mask]
        confidences = confidences[mask]
        
        if len(boxes_yolo) == 0:
            return []
        
        # Преобразование координат из YOLO формата [x_center, y_center, w, h] в [x1, y1, x2, y2]
        # Координаты из модели в пикселях относительно padded изображения 640x640
        target_size = metadata['target_size']
        pad_w, pad_h = metadata['pad']
        scale = metadata['scale']
        
        # Координаты в пикселях модели (включая паддинг)
        x_center = boxes_yolo[:, 0]
        y_center = boxes_yolo[:, 1]
        width = boxes_yolo[:, 2]
        height = boxes_yolo[:, 3]
        
        # Убираем паддинг (переходим к координатам resized изображения)
        x_center = x_center - pad_w
        y_center = y_center - pad_h
        
        # Масштабируем к исходному размеру изображения
        if scale > 0:
            x_center = x_center / scale
            y_center = y_center / scale
            width = width / scale
            height = height / scale
            
        # Преобразование в формат [x1, y1, x2, y2]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Объединение в массив [N, 4]
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Ограничение координат границами изображения
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_shape[0])  # y2
        
        # Фильтрация невалидных боксов (где x2 <= x1 или y2 <= y1)
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_mask]
        class_ids = class_ids[valid_mask]
        class_scores = class_scores[valid_mask]
        confidences = confidences[valid_mask]
        
        if len(boxes) == 0:
            return []
        
        # Применение Non-Maximum Suppression
        keep_indices = nms_per_class(
            boxes,
            class_scores,
            class_ids,
            self.nms_threshold
        )
        
        # Формирование результата
        detections = []
        for idx in keep_indices:
            detections.append([
                float(boxes[idx][0]),  # x1
                float(boxes[idx][1]),  # y1
                float(boxes[idx][2]),  # x2
                float(boxes[idx][3]),  # y2
                int(class_ids[idx]),   # class_id
                float(class_scores[idx])  # confidence
            ])
        
        return detections

