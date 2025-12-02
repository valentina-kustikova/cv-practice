"""
RetinaNet детектор объектов.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os

from .base_detector import BaseDetector
from utils.nms import nms_per_class, compute_iou


class RetinaNetDetector(BaseDetector):
    """Детектор объектов на основе RetinaNet."""
    
    def __init__(self, conf_threshold: float = 0.6, nms_threshold: float = 0.4, 
                 box_expansion: float = 0.0):
        """
        Инициализация RetinaNet детектора.
        
        Args:
            conf_threshold: Порог уверенности (по умолчанию 0.6 для снижения ложных срабатываний)
            nms_threshold: Порог IoU для NMS (по умолчанию 0.4, как у других детекторов)
            box_expansion: Коэффициент расширения боксов (0.0 = без расширения, 0.1 = расширение на 10%)
                          Используется для компенсации неточностей декодирования
        """
        super().__init__(conf_threshold, nms_threshold)
        # RetinaNet работает с изображениями разных размеров, используем 640x640
        self.input_size = (640, 640)
        
        # Параметры якорей
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.anchor_scales = [2**0, 2**(1.0/3.0), 2**(2.0/3.0)]
        self.base_sizes = [32, 64, 128, 256, 512]
        
        # Кэш для якорей (ключ: (stride, H, W), значение: anchors)
        self._anchor_cache = {}
        
        # Флаг для включения логирования производительности
        self._debug_logging = False
        
        # Коэффициент расширения боксов для лучшего обхвата объектов
        self.box_expansion = max(0.0, min(0.5, box_expansion))  # Ограничиваем от 0 до 50%
    
    def load_model(self, weights_path: str, config_path: Optional[str] = None,
                   names_path: Optional[str] = None) -> None:
        """
        Загрузка RetinaNet модели.
        
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
    
    def set_debug_logging(self, enabled: bool = True) -> None:
        """
        Включить/выключить логирование для диагностики производительности.
        
        Args:
            enabled: Включить логирование
        """
        self._debug_logging = enabled
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Предобработка изображения для RetinaNet.
        
        Алгоритм:
        1. Изменение размера с сохранением пропорций
        2. Padding для получения целевого размера
        3. Нормализация ImageNet (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        4. Преобразование в формат NCHW
        
        Args:
            image: Входное изображение в формате BGR
            
        Returns:
            blob: Подготовленный blob
            metadata: Словарь с метаданными для постобработки
        """
        h, w = image.shape[:2]
        target_size = self.input_size[0]  # 640
        
        # Вычисление масштаба с сохранением пропорций
        scale = min(target_size / h, target_size / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize с сохранением пропорций
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Вычисление padding для получения целевого размера
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        pad_h_remainder = target_size - new_h - pad_h
        pad_w_remainder = target_size - new_w - pad_w
        
        # Добавление padding (черный цвет)
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, pad_h_remainder,
            pad_w, pad_w_remainder,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        # Нормализация ImageNet (вычитание среднего, как в Faster R-CNN)
        # mean = [103.939, 116.779, 123.68] (BGR)
        img_normalized = padded.astype(np.float32)
        mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)  # BGR
        img_normalized = img_normalized - mean
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_normalized = img_normalized / std
        
        # Преобразование в формат NCHW
        blob = np.transpose(img_normalized, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        
        # Метаданные для постобработки
        metadata = {
            'scale': scale,
            'pad': (pad_w, pad_h),
            'original_size': (w, h),
            'resized_size': (new_w, new_h),
            'target_size': target_size
        }
        
        return blob, metadata
    
    def postprocess(self, outputs, metadata: dict,
                   original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка выходов RetinaNet.
        
        RetinaNet ONNX обычно возвращает:
        - boxes: [N, 4] - координаты боксов
        - labels: [N] - метки классов
        - scores: [N] - уверенности
        
        Или один тензор [N, 6] где каждая строка: [x1, y1, x2, y2, class_id, score]
        
        Или 10 выходов (FPN): 5 уровней × 2 (boxes + scores)
        
        Args:
            outputs: Выходные тензоры сети
            metadata: Метаданные из preprocess
            original_shape: Исходный размер изображения (height, width)
            
        Returns:
            Список детекций в формате [x1, y1, x2, y2, class_id, confidence]
        """
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        # Проверяем формат выходов
        if len(outputs) == 1:
            # Один тензор - может быть [N, 6] или [1, N, 6]
            output = outputs[0]
            if len(output.shape) == 3:
                output = output[0]  # Убираем batch dimension
            
            return self._postprocess_single_output(output, metadata, original_shape)
        elif len(outputs) == 3:
            # Три тензора: boxes, labels, scores
            return self._postprocess_separate_outputs(outputs, metadata, original_shape)
        elif len(outputs) == 10:
            # FPN: 5 уровней × 2 (boxes + scores)
            return self._postprocess_fpn_outputs(outputs, metadata, original_shape)
        else:
            print(f"WARNING: Неожиданное количество выходов RetinaNet: {len(outputs)}")
            return []
    
    def _postprocess_single_output(self, output: np.ndarray, metadata: dict,
                                   original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка для одного выходного тензора.
        
        Формат: [N, 6] где каждая строка: [x1, y1, x2, y2, class_id, score]
        """
        if len(output) == 0:
            return []
        
        # Извлечение компонентов
        boxes = output[:, :4]  # [N, 4]
        class_ids = output[:, 4].astype(np.int32)  # [N]
        scores = output[:, 5]  # [N]
        
        # Фильтрация по confidence
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            return []
        
        # Преобразование координат из padded изображения в оригинальное
        target_size = metadata['target_size']
        pad_w, pad_h = metadata['pad']
        scale = metadata['scale']
        
        # Координаты относительно padded изображения
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Проверяем формат координат: нормализованные [0, 1] или пиксели
        is_normalized = np.all(x2 <= 1.0) and np.all(y2 <= 1.0) and np.all(x1 >= 0) and np.all(y1 >= 0)
        
        if is_normalized:
            # Нормализованные координаты [0, 1] -> пиксели входного изображения модели
            x1 = x1 * target_size
            y1 = y1 * target_size
            x2 = x2 * target_size
            y2 = y2 * target_size
        
        # Теперь координаты в пикселях входного изображения модели (640x640 с padding)
        # Убираем padding (переходим к координатам resized изображения)
        x1 = x1 - pad_w
        y1 = y1 - pad_h
        x2 = x2 - pad_w
        y2 = y2 - pad_h
        
        # Масштабируем обратно к оригинальному размеру изображения
        if scale > 0:
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
        
        # Применяем расширение боксов для лучшего обхвата объектов
        if self.box_expansion > 0:
            box_widths = x2 - x1
            box_heights = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            expanded_widths = box_widths * (1 + self.box_expansion)
            expanded_heights = box_heights * (1 + self.box_expansion)
            x1 = center_x - expanded_widths / 2
            y1 = center_y - expanded_heights / 2
            x2 = center_x + expanded_widths / 2
            y2 = center_y + expanded_heights / 2
        
        # Ограничение координат границами изображения
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Фильтрация невалидных боксов
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_mask]
        class_ids = class_ids[valid_mask]
        scores = scores[valid_mask]
        
        if len(boxes) == 0:
            return []
        
        # Применение NMS
        keep_indices = nms_per_class(
            boxes,
            scores,
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
                float(scores[idx])     # confidence
            ])
        
        # Применяем фильтр дубликатов для удаления боксов с разной ориентацией
        if len(detections) > 0:
            det_boxes = np.array([d[:4] for d in detections])
            det_scores = np.array([d[5] for d in detections])
            det_class_ids = np.array([d[4] for d in detections], dtype=np.int32)
            
            keep_indices_dup = self._filter_duplicate_boxes(
                det_boxes,
                det_scores,
                det_class_ids,
                center_threshold=0.08,
                iou_threshold=0.3
            )
            
            detections = [detections[idx] for idx in keep_indices_dup]
        
        return detections
    
    def _postprocess_separate_outputs(self, outputs: List[np.ndarray], metadata: dict,
                                      original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка для раздельных выходов (boxes, labels, scores).
        """
        boxes_tensor = outputs[0]
        labels_tensor = outputs[1]
        scores_tensor = outputs[2]
        
        # Убираем batch dimension если есть
        if len(boxes_tensor.shape) == 3:
            boxes_tensor = boxes_tensor[0]
        if len(labels_tensor.shape) == 2:
            labels_tensor = labels_tensor[0]
        if len(scores_tensor.shape) == 2:
            scores_tensor = scores_tensor[0]
        
        # Фильтрация по confidence
        mask = scores_tensor >= self.conf_threshold
        boxes = boxes_tensor[mask]
        class_ids = labels_tensor[mask].astype(np.int32)
        scores = scores_tensor[mask]
        
        if len(boxes) == 0:
            return []
        
        # Преобразование координат (аналогично _postprocess_single_output)
        target_size = metadata['target_size']
        pad_w, pad_h = metadata['pad']
        scale = metadata['scale']
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Проверяем формат координат: нормализованные [0, 1] или пиксели
        is_normalized = np.all(x2 <= 1.0) and np.all(y2 <= 1.0) and np.all(x1 >= 0) and np.all(y1 >= 0)
        
        if is_normalized:
            # Нормализованные координаты [0, 1] -> пиксели входного изображения модели
            x1 = x1 * target_size
            y1 = y1 * target_size
            x2 = x2 * target_size
            y2 = y2 * target_size
        
        # Теперь координаты в пикселях входного изображения модели (640x640 с padding)
        # Убираем padding (переходим к координатам resized изображения)
        x1 = x1 - pad_w
        y1 = y1 - pad_h
        x2 = x2 - pad_w
        y2 = y2 - pad_h
        
        # Масштабируем обратно к оригинальному размеру изображения
        if scale > 0:
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
        
        # Применяем расширение боксов для лучшего обхвата объектов
        if self.box_expansion > 0:
            box_widths = x2 - x1
            box_heights = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            expanded_widths = box_widths * (1 + self.box_expansion)
            expanded_heights = box_heights * (1 + self.box_expansion)
            x1 = center_x - expanded_widths / 2
            y1 = center_y - expanded_heights / 2
            x2 = center_x + expanded_widths / 2
            y2 = center_y + expanded_heights / 2
        
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Фильтрация невалидных боксов
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_mask]
        class_ids = class_ids[valid_mask]
        scores = scores[valid_mask]
        
        if len(boxes) == 0:
            return []
        
        # Применение NMS
        keep_indices = nms_per_class(
            boxes,
            scores,
            class_ids,
            self.nms_threshold
        )
        
        # Формирование результата
        detections = []
        for idx in keep_indices:
            detections.append([
                float(boxes[idx][0]),
                float(boxes[idx][1]),
                float(boxes[idx][2]),
                float(boxes[idx][3]),
                int(class_ids[idx]),
                float(scores[idx])
            ])
        
        # Применяем фильтр дубликатов для удаления боксов с разной ориентацией
        if len(detections) > 0:
            det_boxes = np.array([d[:4] for d in detections])
            det_scores = np.array([d[5] for d in detections])
            det_class_ids = np.array([d[4] for d in detections], dtype=np.int32)
            
            keep_indices_dup = self._filter_duplicate_boxes(
                det_boxes,
                det_scores,
                det_class_ids,
                center_threshold=0.08,
                iou_threshold=0.3
            )
            
            detections = [detections[idx] for idx in keep_indices_dup]
        
        return detections
    
    def _postprocess_fpn_outputs(self, outputs: List[np.ndarray], metadata: dict,
                                  original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Постобработка для FPN выходов (Feature Pyramid Network).
        
        RetinaNet с FPN возвращает 10 выходов:
        - Outputs 0-5: scores [B, 720, H, W] где 720 = 9 anchors x 80 классов
        - Outputs 6-9: boxes [B, 36, H, W] где 36 = 9 anchors x 4 координаты
        
        Но порядок может быть смешанным, поэтому разделяем по размеру канала.
        """
        # Разделяем выходы на boxes (36 каналов) и scores (720 каналов)
        boxes_outputs = []
        scores_outputs = []
        
        for out in outputs:
            if len(out.shape) == 4:
                batch, channels, h, w = out.shape
                if channels == 36:  # Boxes: 9 anchors x 4
                    boxes_outputs.append(out[0])  # Убираем batch dimension
                elif channels == 720:  # Scores: 9 anchors x 80 классов
                    scores_outputs.append(out[0])
        
        # Сортируем по размеру feature map (от большого к маленькому)
        boxes_outputs.sort(key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        scores_outputs.sort(key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        
        # ДИАГНОСТИКА: Формат выходов модели
        if self._debug_logging:
            print("\n" + "="*60)
            print("ДИАГНОСТИКА: Формат выходов модели")
            print("="*60)
            print(f"Всего выходов: {len(outputs)}")
            print(f"Boxes outputs: {len(boxes_outputs)}")
            print(f"Scores outputs: {len(scores_outputs)}")
            
            for i, out in enumerate(outputs):
                if len(out.shape) == 4:
                    batch, channels, h, w = out.shape
                    print(f"\nВыход {i}:")
                    print(f"  Форма: {out.shape} (batch={batch}, channels={channels}, H={h}, W={w})")
                    print(f"  Диапазон значений: [{out.min():.4f}, {out.max():.4f}]")
                    print(f"  Среднее: {out.mean():.4f}, Std: {out.std():.4f}")
                    if channels == 36:
                        print(f"  Тип: BOXES (9 anchors x 4 координаты)")
                    elif channels == 720:
                        print(f"  Тип: SCORES (9 anchors x 80 классов)")
                    else:
                        print(f"  Тип: НЕИЗВЕСТНЫЙ (ожидалось 36 или 720 каналов)")
            
            print(f"\nПроверка соответствия формату:")
            print(f"  Ожидается: {len(boxes_outputs)} уровней boxes и {len(scores_outputs)} уровней scores")
            if len(boxes_outputs) != len(scores_outputs):
                print(f"  [WARN] Несоответствие количества уровней!")
            
            for i, (box_out, score_out) in enumerate(zip(boxes_outputs, scores_outputs)):
                box_shape = box_out.shape
                score_shape = score_out.shape
                print(f"\nУровень FPN {i}:")
                print(f"  Boxes: {box_shape} (channels={box_shape[0]}, H={box_shape[1]}, W={box_shape[2]})")
                print(f"  Scores: {score_shape} (channels={score_shape[0]}, H={score_shape[1]}, W={score_shape[2]})")
                print(f"  Boxes диапазон: [{box_out.min():.4f}, {box_out.max():.4f}]")
                print(f"  Scores диапазон: [{score_out.min():.4f}, {score_out.max():.4f}]")
                if box_shape[1] != score_shape[1] or box_shape[2] != score_shape[2]:
                    print(f"  [WARN] Размеры feature map не совпадают!")
            print("="*60 + "\n")
        
        all_detections = []
        
        # Обрабатываем каждый уровень FPN
        if self._debug_logging:
            print(f"Processing {len(boxes_outputs)} FPN levels")
        
        for i, (boxes_out, scores_out) in enumerate(zip(boxes_outputs, scores_outputs)):
            level_detections = self._process_fpn_level(
                boxes_out, scores_out, metadata, original_shape
            )
            all_detections.extend(level_detections)
        
        if self._debug_logging:
            print(f"Total detections before filtering: {len(all_detections)}")
        
        if len(all_detections) == 0:
            return []
        
        # Применяем NMS ко всем детекциям
        boxes = np.array([d[:4] for d in all_detections])
        scores = np.array([d[5] for d in all_detections])
        class_ids = np.array([d[4] for d in all_detections], dtype=np.int32)
        
        # ИСПРАВЛЕНИЕ: Убрано каскадное повышение порога
        # Все детекции уже прошли фильтрацию по self.conf_threshold в _process_fpn_level
        # Применяем NMS с обычным порогом
        final_indices = nms_per_class(boxes, scores, class_ids, self.nms_threshold)
        
        if self._debug_logging:
            print(f"After NMS: {len(final_indices)} final detections")
        
        final_detections = [all_detections[idx] for idx in final_indices]
        
        # ИСПРАВЛЕНИЕ: Убрана дополнительная фильтрация после NMS
        # Все детекции уже соответствуют порогу уверенности
        
        # ИСПРАВЛЕНИЕ: Применяем фильтр дубликатов для удаления боксов с разной ориентацией
        # (вертикальный/горизонтальный), которые обрамляют один объект
        if len(final_detections) > 0:
            final_boxes = np.array([d[:4] for d in final_detections])
            final_scores = np.array([d[5] for d in final_detections])
            final_class_ids = np.array([d[4] for d in final_detections], dtype=np.int32)
            
            # Применяем фильтр дубликатов
            keep_indices = self._filter_duplicate_boxes(
                final_boxes,
                final_scores,
                final_class_ids,
                center_threshold=0.08,  # Порог близости центров (8% от среднего размера)
                iou_threshold=0.3  # Порог IoU для определения дубликатов
            )
            
            final_detections = [final_detections[idx] for idx in keep_indices]
            
            if self._debug_logging:
                print(f"After duplicate filtering: {len(final_detections)} final detections")
        
        return final_detections
    
    def _process_fpn_level(self, boxes_out: np.ndarray, scores_out: np.ndarray,
                           metadata: dict, original_shape: Tuple[int, int]) -> List[List[float]]:
        """
        Обработка одного уровня FPN с оптимизацией производительности.
        
        Args:
        boxes_out: [36, H, W] - предсказания боксов (36 = 9 anchors x 4)
        scores_out: [720, H, W] - предсказания классов (720 = 9 anchors x 80 классов)
        """
        # Формат: [C, H, W] -> переводим в [H, W, C]
        boxes_out = np.transpose(boxes_out, (1, 2, 0))  # [H, W, 36]
        scores_out = np.transpose(scores_out, (1, 2, 0))  # [H, W, 720]
        
        H, W = boxes_out.shape[:2]
        
        # Определяем количество anchors и классов
        num_box_values = boxes_out.shape[2]  # 36
        num_score_values = scores_out.shape[2]  # 720
        
        # Обычно A=9 anchors, num_classes=80 (COCO)
        num_anchors = num_box_values // 4  # 9
        num_classes = num_score_values // num_anchors  # 80
        
        # Reshape
        boxes_out = boxes_out.reshape(H, W, num_anchors, 4)  # [H, W, 9, 4]
        scores_out = scores_out.reshape(H, W, num_anchors, num_classes)  # [H, W, 9, 80]
        
        # Автоматическое определение формата scores (логиты или вероятности)
        score_sample = scores_out.flatten()[:1000]  # Берем выборку для анализа
        score_min = score_sample.min()
        score_max = score_sample.max()
        score_mean = score_sample.mean()
        
        # Определяем, нужен ли sigmoid
        # Если значения уже в [0, 1] и среднее > 0 - вероятности
        # Иначе - логиты, нужен sigmoid
        need_sigmoid = not (score_min >= -0.1 and score_max <= 1.1 and score_mean > 0.01)
        
        # Векторизованная обработка всех scores сразу
        if need_sigmoid:
            # Применяем sigmoid для получения вероятностей из логитов
            class_probs_all = 1 / (1 + np.exp(-np.clip(scores_out, -500, 500)))
        else:
            # Уже вероятности, просто обрезаем
            class_probs_all = np.clip(scores_out, 0, 1)
        
        # ИСПРАВЛЕНИЕ: Обрабатываем все классы независимо вместо argmax
        # Это стандартная схема RetinaNet - каждый якорь может иметь несколько активных классов
        # Формируем маску для всех пар (якорь, класс) с уверенностью >= порога
        class_mask = class_probs_all >= self.conf_threshold  # [H, W, 9, 80]
        
        # Находим все валидные пары (y, x, anchor, class)
        valid_positions = np.argwhere(class_mask)  # [N, 4] где N - количество валидных пар
        
        if len(valid_positions) == 0:
            return []
        
        # Извлекаем координаты и уверенности для всех валидных пар
        valid_y = valid_positions[:, 0]
        valid_x = valid_positions[:, 1]
        valid_a = valid_positions[:, 2]
        valid_c = valid_positions[:, 3]
        
        # Уверенности для валидных пар
        valid_confidences = class_probs_all[valid_y, valid_x, valid_a, valid_c]
        
        initial_candidates = len(valid_positions)
        
        # Лимитируем количество кандидатов на уровне FPN
        # Делаем зависимым от размера feature map: больше feature map = больше кандидатов
        base_max_candidates = 100
        feature_map_size = H * W
        # Масштабируем: для больших feature maps (например, 80x80) даем больше кандидатов
        max_candidates_per_level = int(base_max_candidates * (1 + feature_map_size / 1000))
        limited_indices = None

        # Вычисление stride
        # Более точное вычисление: stride = размер входного изображения / размер feature map
        stride = self.input_size[0] / W
        if stride < 1:
            stride = 1
        
        # Округляем stride до целого для генерации якорей
        stride_int = int(round(stride))
        
        if self._debug_logging:
            print(f"  FPN Level: shape={H}x{W}, stride={stride:.1f}, "
                  f"initial_candidates={initial_candidates}, "
                  f"need_sigmoid={need_sigmoid}")
        
        # Генерация якорей
        anchors = self._generate_anchors(stride_int, (H, W))
        
        # ДИАГНОСТИКА: Генерация якорей
        if self._debug_logging:
            print(f"\n  ДИАГНОСТИКА: Генерация якорей для уровня FPN")
            print(f"  Stride: {stride:.2f} (округлен до {stride_int})")
            print(f"  Размер feature map: H={H}, W={W}")
            print(f"  Base size: {stride_int * 4}")
            print(f"  Форма якорей: {anchors.shape} (ожидается [H, W, 9, 4])")
            
            # Проверка количества якорей на ячейку
            num_anchors_per_cell = anchors.shape[2]
            print(f"  Якорей на ячейку: {num_anchors_per_cell} (ожидается 9)")
            if num_anchors_per_cell != 9:
                print(f"  [WARN] Неожиданное количество якорей!")
            
            # Статистика размеров якорей
            anchor_widths = anchors[:, :, :, 2] - anchors[:, :, :, 0]  # x2 - x1
            anchor_heights = anchors[:, :, :, 3] - anchors[:, :, :, 1]  # y2 - y1
            print(f"  Размеры якорей:")
            print(f"    Ширина: min={anchor_widths.min():.2f}, max={anchor_widths.max():.2f}, mean={anchor_widths.mean():.2f}")
            print(f"    Высота: min={anchor_heights.min():.2f}, max={anchor_heights.max():.2f}, mean={anchor_heights.mean():.2f}")
            
            # Примеры координат якорей (первые 3 якоря первой ячейки)
            print(f"  Примеры якорей (первые 3 якоря первой ячейки [0,0]):")
            for a_idx in range(min(3, num_anchors_per_cell)):
                anchor = anchors[0, 0, a_idx, :]
                center_x = (anchor[0] + anchor[2]) / 2
                center_y = (anchor[1] + anchor[3]) / 2
                width = anchor[2] - anchor[0]
                height = anchor[3] - anchor[1]
                print(f"    Якорь {a_idx}: x1={anchor[0]:.2f}, y1={anchor[1]:.2f}, "
                      f"x2={anchor[2]:.2f}, y2={anchor[3]:.2f}")
                print(f"      Центр: ({center_x:.2f}, {center_y:.2f}), Размер: {width:.2f}x{height:.2f}")
            
            # Проверка, что якори находятся в пределах изображения модели (640x640)
            target_size = self.input_size[0]
            anchors_out_of_bounds = np.any((anchors < 0) | (anchors > target_size))
            if anchors_out_of_bounds:
                print(f"  [WARN] Некоторые якори выходят за пределы изображения модели ({target_size}x{target_size})")
                out_of_bounds_count = np.sum((anchors < 0) | (anchors > target_size))
                print(f"  Количество якорей вне границ: {out_of_bounds_count}")
            else:
                print(f"  OK: Все якори находятся в пределах изображения модели ({target_size}x{target_size})")
        
        # Диагностика формата боксов (только при включенном логировании)
        if self._debug_logging:
            sample_box = boxes_out[0, 0, 0, :]  # Первый бокс первого якоря
            print(f"  Raw box values (before decode): {sample_box}")
            print(f"  Raw box range: [{boxes_out.min():.2f}, {boxes_out.max():.2f}]")
            print(f"  Raw box mean: {boxes_out.mean():.2f}, std: {boxes_out.std():.2f}")

        # Фильтрация якорей, которые попадают в область паддинга
        pad_w_left, pad_h_top = metadata['pad']
        resized_w, resized_h = metadata['resized_size']
        target_size = metadata['target_size']
        pad_w_right = target_size - resized_w - pad_w_left
        pad_h_bottom = target_size - resized_h - pad_h_top
        valid_x_min = pad_w_left
        valid_x_max = target_size - pad_w_right
        valid_y_min = pad_h_top
        valid_y_max = target_size - pad_h_bottom
        
        centers_x = (anchors[..., 0] + anchors[..., 2]) / 2.0
        centers_y = (anchors[..., 1] + anchors[..., 3]) / 2.0
        
        # Фильтрация якорей, которые попадают в область паддинга
        valid_area_mask = (
            (centers_x >= valid_x_min) & (centers_x <= valid_x_max) &
            (centers_y >= valid_y_min) & (centers_y <= valid_y_max)
        )
        
        # Применяем маску валидной области к позициям
        valid_area_for_positions = valid_area_mask[valid_y, valid_x, valid_a]
        valid_positions = valid_positions[valid_area_for_positions]
        valid_confidences = valid_confidences[valid_area_for_positions]
        
        # Обновляем координаты после фильтрации
        if len(valid_positions) == 0:
            return []
        
        valid_y = valid_positions[:, 0]
        valid_x = valid_positions[:, 1]
        valid_a = valid_positions[:, 2]
        valid_c = valid_positions[:, 3]
        
        after_area_candidates = len(valid_positions)
        
        if self._debug_logging:
            removed = initial_candidates - after_area_candidates
            print(f"  Маска валидной области: осталось {after_area_candidates}, удалено {removed}")
        
        # Лимитирование количества кандидатов по уверенности
        if after_area_candidates > max_candidates_per_level:
            # Сортируем по уверенности и берем топ-K
            top_indices = np.argsort(valid_confidences)[::-1][:max_candidates_per_level]
            valid_positions = valid_positions[top_indices]
            valid_confidences = valid_confidences[top_indices]
            
            # Обновляем координаты
            valid_y = valid_positions[:, 0]
            valid_x = valid_positions[:, 1]
            valid_a = valid_positions[:, 2]
            valid_c = valid_positions[:, 3]
        
        after_limit = len(valid_positions)
        
        if self._debug_logging:
            print(f"  Кандидатов после лимита: {after_limit}")
        
        # Извлекаем валидные deltas и anchors
        valid_deltas = boxes_out[valid_y, valid_x, valid_a, :]  # [N, 4]
        valid_anchors = anchors[valid_y, valid_x, valid_a, :]  # [N, 4]
        
        # ДИАГНОСТИКА: Декодирование боксов
        if self._debug_logging:
            print(f"\n  ДИАГНОСТИКА: Декодирование боксов")
            print(f"  Количество валидных кандидатов: {len(valid_deltas)}")
            
            # Анализ deltas ДО декодирования
            print(f"  Deltas (до декодирования):")
            print(f"    Диапазон: [{valid_deltas.min():.4f}, {valid_deltas.max():.4f}]")
            print(f"    Среднее: {valid_deltas.mean():.4f}, Std: {valid_deltas.std():.4f}")
            print(f"    По каналам:")
            for i in range(4):
                channel = valid_deltas[:, i]
                print(f"      Канал {i}: min={channel.min():.4f}, max={channel.max():.4f}, mean={channel.mean():.4f}")
            
            # Определение формата deltas
            delta_max = np.abs(valid_deltas).max()
            delta_mean = np.abs(valid_deltas).mean()
            if delta_max > 100:
                if delta_max < 1000 and np.all(valid_deltas[:, 2] > valid_deltas[:, 0]) and np.all(valid_deltas[:, 3] > valid_deltas[:, 1]):
                    delta_format = "Абсолютные координаты (x1, y1, x2, y2)"
                else:
                    delta_format = "Абсолютные координаты (x, y, w, h)"
            else:
                delta_format = "Смещения относительно якорей (dx, dy, dw, dh)"
            print(f"    Определенный формат: {delta_format}")
            
            # Примеры: anchor → delta → decoded box
            num_examples = min(3, len(valid_deltas))
            print(f"  Примеры преобразования (первые {num_examples}):")
            for i in range(num_examples):
                anchor = valid_anchors[i]
                delta = valid_deltas[i]
                anchor_center_x = (anchor[0] + anchor[2]) / 2
                anchor_center_y = (anchor[1] + anchor[3]) / 2
                anchor_w = anchor[2] - anchor[0]
                anchor_h = anchor[3] - anchor[1]
                print(f"    Пример {i+1}:")
                print(f"      Якорь: center=({anchor_center_x:.2f}, {anchor_center_y:.2f}), size={anchor_w:.2f}x{anchor_h:.2f}")
                print(f"      Delta: [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}, {delta[3]:.4f}]")
        
        # Декодируем только валидные боксы (оптимизированная версия)
        decoded_boxes_valid = self._decode_boxes_fast(valid_anchors, valid_deltas)
        
        # ИСПРАВЛЕНИЕ: Используем уже извлеченные confidence и class_id из независимой обработки классов
        # valid_confidences и valid_c уже содержат правильные значения для каждой пары (якорь, класс)
        
        # Диагностика после декодирования
        if self._debug_logging and len(decoded_boxes_valid) > 0:
            print(f"  Decoded boxes (после декодирования):")
            print(f"    Диапазон: [{decoded_boxes_valid.min():.2f}, {decoded_boxes_valid.max():.2f}]")
            is_normalized = decoded_boxes_valid.max() <= 1.0
            print(f"    Формат: {'normalized [0,1]' if is_normalized else 'pixels'}")
            
            # Проверка валидности декодированных боксов
            x1_decoded = decoded_boxes_valid[:, 0]
            y1_decoded = decoded_boxes_valid[:, 1]
            x2_decoded = decoded_boxes_valid[:, 2]
            y2_decoded = decoded_boxes_valid[:, 3]
            
            invalid_boxes = (x2_decoded <= x1_decoded) | (y2_decoded <= y1_decoded)
            invalid_count = np.sum(invalid_boxes)
            if invalid_count > 0:
                print(f"    [WARN] {invalid_count} невалидных боксов (x2<=x1 или y2<=y1)")
            else:
                print(f"    OK: Все боксы валидны (x2>x1 и y2>y1)")
            
            # Статистика размеров декодированных боксов
            widths_decoded = x2_decoded - x1_decoded
            heights_decoded = y2_decoded - y1_decoded
            print(f"    Размеры декодированных боксов:")
            print(f"      Ширина: min={widths_decoded.min():.2f}, max={widths_decoded.max():.2f}, mean={widths_decoded.mean():.2f}")
            print(f"      Высота: min={heights_decoded.min():.2f}, max={heights_decoded.max():.2f}, mean={heights_decoded.mean():.2f}")
            
            # Показываем примеры декодированных боксов
            num_examples = min(3, len(decoded_boxes_valid))
            print(f"    Примеры декодированных боксов (первые {num_examples}):")
            for i in range(num_examples):
                box = decoded_boxes_valid[i]
                print(f"      Бокс {i+1}: x1={box[0]:.2f}, y1={box[1]:.2f}, x2={box[2]:.2f}, y2={box[3]:.2f}, "
                      f"size={box[2]-box[0]:.2f}x{box[3]-box[1]:.2f}")
        
        # Векторизованная обработка всех валидных боксов
        min_box_size = 4.0
        pad_w, pad_h = metadata['pad']
        scale = metadata['scale']
        target_size = metadata['target_size']
        
        # Извлекаем координаты
        x1 = decoded_boxes_valid[:, 0]
        y1 = decoded_boxes_valid[:, 1]
        x2 = decoded_boxes_valid[:, 2]
        y2 = decoded_boxes_valid[:, 3]
        
        # Исправление перевернутых боксов (векторизованно)
        x1, x2 = np.minimum(x1, x2), np.maximum(x1, x2)
        y1, y2 = np.minimum(y1, y2), np.maximum(y1, y2)
        
        # Проверка формата координат (для всех сразу)
        # Если большинство координат <= 1.0, это нормализованные координаты
        is_normalized = np.mean(decoded_boxes_valid) < 10.0
        
        if is_normalized:
            # Нормализованные координаты -> пиксели
            x1 = x1 * target_size
            y1 = y1 * target_size
            x2 = x2 * target_size
            y2 = y2 * target_size
        
        # Проверка минимального размера (векторизованно)
        box_widths = x2 - x1
        box_heights = y2 - y1
        size_mask = (box_widths >= min_box_size) & (box_heights >= min_box_size)
        
        # Применяем маску
        x1 = x1[size_mask]
        y1 = y1[size_mask]
        x2 = x2[size_mask]
        y2 = y2[size_mask]
        valid_confidences = valid_confidences[size_mask]
        valid_c = valid_c[size_mask]  # Классы для валидных боксов
        box_widths = box_widths[size_mask]
        box_heights = box_heights[size_mask]
        
        if len(x1) == 0:
            return []
        
        # Применяем расширение боксов (векторизованно)
        if self.box_expansion > 0:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            expanded_widths = box_widths * (1 + self.box_expansion)
            expanded_heights = box_heights * (1 + self.box_expansion)
            x1 = center_x - expanded_widths / 2
            y1 = center_y - expanded_heights / 2
            x2 = center_x + expanded_widths / 2
            y2 = center_y + expanded_heights / 2
        
        # Преобразование координат (векторизованно)
        if scale > 0:
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
        else:
            x1 = x1 - pad_w
            y1 = y1 - pad_h
            x2 = x2 - pad_w
            y2 = y2 - pad_h
        
        # Повторное исправление перевернутых боксов
        x1, x2 = np.minimum(x1, x2), np.maximum(x1, x2)
        y1, y2 = np.minimum(y1, y2), np.maximum(y1, y2)
        
        # Ограничение координат (векторизованно)
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])
        
        # Финальная фильтрация валидных боксов
        # Более строгая фильтрация: минимальный размер бокса увеличен для RetinaNet
        # чтобы убрать слишком маленькие ложные срабатывания
        min_box_size = 8.0  # Увеличено с 2.0 до 8.0 для фильтрации мелких ложных срабатываний
        valid_mask = (x2 > x1) & (y2 > y1) & ((x2 - x1) >= min_box_size) & ((y2 - y1) >= min_box_size)
        
        x1 = x1[valid_mask]
        y1 = y1[valid_mask]
        x2 = x2[valid_mask]
        y2 = y2[valid_mask]
        valid_confidences = valid_confidences[valid_mask]
        valid_c = valid_c[valid_mask]
        
        if len(x1) == 0:
            return []
        
        # Формирование результата (векторизованно)
        detections = np.stack([
            x1.astype(np.float32),
            y1.astype(np.float32),
            x2.astype(np.float32),
            y2.astype(np.float32),
            valid_c.astype(np.int32),
            valid_confidences.astype(np.float32)
        ], axis=1).tolist()
        
        # Преобразуем в список списков
        detections = [[float(d[0]), float(d[1]), float(d[2]), float(d[3]), int(d[4]), float(d[5])] for d in detections]
        
        if self._debug_logging:
            print(f"    Final detections from this level: {len(detections)}")
        
        return detections

    def _generate_anchors(self, stride: int, shape: Tuple[int, int]) -> np.ndarray:
        """
        Генерация якорей для одного уровня пирамиды с кэшированием.
        
        Args:
            stride: Шаг сетки (например, 8, 16, 32...)
            shape: Размер сетки (H, W)
            
        Returns:
            Anchors: [H, W, 9, 4] (x1, y1, x2, y2)
        """
        H, W = shape
        
        # Проверяем кэш
        cache_key = (stride, H, W)
        if cache_key in self._anchor_cache:
            return self._anchor_cache[cache_key]
        
        # В RetinaNet base_size обычно равен stride
        # Но для совместимости с разными моделями используем stride * 4
        base_size = stride * 4
        
        scales = np.array(self.anchor_scales)
        ratios = np.array(self.anchor_ratios)
        
        num_anchors = len(scales) * len(ratios)
        
        # Генерация базовых якорей для одной ячейки
        # В RetinaNet формула для размеров якорей:
        # w = base_size * scale * sqrt(ratio)
        # h = base_size * scale / sqrt(ratio)
        # Это сохраняет площадь примерно постоянной при разных соотношениях сторон
        
        anchor_dims = []
        for scale in scales:
            for ratio in ratios:
                # Более точная формула для RetinaNet
                w = base_size * scale * np.sqrt(ratio)
                h = base_size * scale / np.sqrt(ratio)
                anchor_dims.append([w, h])
        
        anchor_dims = np.array(anchor_dims)  # [9, 2]
        
        # Сетка центров якорей
        # В RetinaNet центры обычно находятся в центре каждой ячейки feature map
        # Для feature map размером HxW с stride S, центры находятся в:
        # x = (i + 0.5) * S, где i = 0, 1, ..., W-1
        # y = (j + 0.5) * S, где j = 0, 1, ..., H-1
        
        # Альтернативный вариант: центры в начале ячейки + offset
        # Используем более стандартную формулу для RetinaNet
        shift_x = (np.arange(0, W) + 0.5) * stride
        shift_y = (np.arange(0, H) + 0.5) * stride
        
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # [H, W]
        
        # [H, W, 9, 2] - центры для каждого якоря
        centers_x = np.expand_dims(shift_x, axis=-1).repeat(num_anchors, axis=-1)
        centers_y = np.expand_dims(shift_y, axis=-1).repeat(num_anchors, axis=-1)
        
        # [H, W, 9, 2] - размеры для каждого якоря
        # Сначала расширяем anchor_dims до [1, 1, 9, 2] и тайлим
        anchor_wh = anchor_dims.reshape(1, 1, num_anchors, 2)
        anchor_wh = np.tile(anchor_wh, (H, W, 1, 1))
        
        # Координаты x1, y1, x2, y2
        # x1 = cx - w/2
        anchors = np.zeros((H, W, num_anchors, 4))
        anchors[..., 0] = centers_x - anchor_wh[..., 0] / 2  # x1
        anchors[..., 1] = centers_y - anchor_wh[..., 1] / 2  # y1
        anchors[..., 2] = centers_x + anchor_wh[..., 0] / 2  # x2
        anchors[..., 3] = centers_y + anchor_wh[..., 1] / 2  # y2
        
        # ИСПРАВЛЕНИЕ: Ограничиваем якори границами изображения модели
        # Якори не должны выходить за пределы [0, target_size]
        target_size = self.input_size[0]  # 640
        anchors[..., 0] = np.clip(anchors[..., 0], 0, target_size)  # x1
        anchors[..., 1] = np.clip(anchors[..., 1], 0, target_size)  # y1
        anchors[..., 2] = np.clip(anchors[..., 2], 0, target_size)  # x2
        anchors[..., 3] = np.clip(anchors[..., 3], 0, target_size)  # y2
        
        # Удаляем якори с нулевой площадью (которые были полностью обрезаны)
        anchor_widths = anchors[..., 2] - anchors[..., 0]
        anchor_heights = anchors[..., 3] - anchors[..., 1]
        # Оставляем только якори с минимальным размером (например, 1 пиксель)
        valid_anchors_mask = (anchor_widths >= 1.0) & (anchor_heights >= 1.0)
        # Если якорь стал невалидным, устанавливаем его в центр ячейки с минимальным размером
        invalid_mask = ~valid_anchors_mask
        if np.any(invalid_mask):
            # Для невалидных якорей устанавливаем минимальный размер в центре ячейки
            anchors[invalid_mask, 0] = np.maximum(0, centers_x[invalid_mask] - 0.5)
            anchors[invalid_mask, 1] = np.maximum(0, centers_y[invalid_mask] - 0.5)
            anchors[invalid_mask, 2] = np.minimum(target_size, centers_x[invalid_mask] + 0.5)
            anchors[invalid_mask, 3] = np.minimum(target_size, centers_y[invalid_mask] + 0.5)
        
        # ДИАГНОСТИКА: Параметры генерации якорей (в методе _generate_anchors)
        if self._debug_logging:
            print(f"    [Генерация якорей] Stride={stride}, Base_size={base_size}")
            print(f"    Scales: {scales}")
            print(f"    Ratios: {ratios}")
            print(f"    Количество якорей на ячейку: {num_anchors}")
            print(f"    Размеры якорей (w, h):")
            for i, dims in enumerate(anchor_dims):
                print(f"      Якорь {i}: w={dims[0]:.2f}, h={dims[1]:.2f}, aspect_ratio={dims[0]/dims[1]:.2f}")
        
        # Сохраняем в кэш
        self._anchor_cache[cache_key] = anchors
        
        return anchors

    def _decode_boxes_fast(self, anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Быстрое декодирование предсказаний для валидных боксов только.
        
        Args:
            anchors: [N, 4] (x1, y1, x2, y2) - только валидные якори
            deltas: [N, 4] (dx, dy, dw, dh) или (x, y, w, h) - только валидные предсказания
            
        Returns:
            Decoded boxes: [N, 4] (x1, y1, x2, y2)
        """
        # Размеры и центры якорей
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        # Быстрая проверка формата (только для первого элемента)
        delta_max = np.abs(deltas).max()
        
        if delta_max > 100:
            # Абсолютные координаты
            if delta_max < 1000 and np.all(deltas[:, 2] > deltas[:, 0]) and np.all(deltas[:, 3] > deltas[:, 1]):
                return deltas.copy()
            else:
                # Формат (x, y, w, h)
                x = deltas[:, 0]
                y = deltas[:, 1]
                w = deltas[:, 2]
                h = deltas[:, 3]
                pred_boxes = np.zeros_like(deltas)
                pred_boxes[:, 0] = x - 0.5 * w
                pred_boxes[:, 1] = y - 0.5 * h
                pred_boxes[:, 2] = x + 0.5 * w
                pred_boxes[:, 3] = y + 0.5 * h
                return pred_boxes
        else:
            # Стандартная параметризация RetinaNet
            dx = deltas[:, 0]
            dy = deltas[:, 1]
            dw = deltas[:, 2]
            dh = deltas[:, 3]
            
            widths_safe = np.maximum(widths, 1.0)
            heights_safe = np.maximum(heights, 1.0)
            
            pred_ctr_x = dx * widths_safe + ctr_x
            pred_ctr_y = dy * heights_safe + ctr_y
            
            dw_clipped = np.clip(dw, -10, 10)
            dh_clipped = np.clip(dh, -10, 10)
            
            pred_w = np.exp(dw_clipped) * widths_safe
            pred_h = np.exp(dh_clipped) * heights_safe
            
            pred_boxes = np.zeros_like(deltas)
            pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
            pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
            pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
            pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
            
            return pred_boxes
    
    def _decode_boxes(self, anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Декодирование предсказаний (смещений) относительно якорей.
        
        RetinaNet использует стандартную параметризацию:
        - dx, dy: смещение центра относительно якоря (нормализованное на размеры якоря)
        - dw, dh: логарифмическое изменение размеров
        
        Args:
            anchors: [H, W, A, 4] (x1, y1, x2, y2)
            deltas: [H, W, A, 4] (dx, dy, dw, dh) или (x, y, w, h)
            
        Returns:
            Decoded boxes: [H, W, A, 4] (x1, y1, x2, y2)
        """
        # Размеры и центры якорей
        widths = anchors[..., 2] - anchors[..., 0]
        heights = anchors[..., 3] - anchors[..., 1]
        ctr_x = anchors[..., 0] + 0.5 * widths
        ctr_y = anchors[..., 1] + 0.5 * heights
        
        # Проверяем диапазон значений deltas для определения формата
        # Если значения очень большие (>100), это могут быть абсолютные координаты
        # Если значения маленькие (обычно [-10, 10]), это смещения
        delta_max = np.abs(deltas).max()
        delta_min = np.abs(deltas).min()
        delta_mean = np.abs(deltas).mean()
        
        # Более точное определение формата
        # Если большинство значений в диапазоне [0, 640] и среднее > 50 - вероятно абсолютные координаты
        # Если значения в диапазоне [-10, 10] - смещения
        if delta_max > 100 or (delta_mean > 50 and delta_max < 1000):
            # Возможно, это абсолютные координаты (x, y, w, h) или (x1, y1, x2, y2)
            # Проверяем, если это уже x1, y1, x2, y2
            if delta_max < 1000 and np.all(deltas[..., 2] > deltas[..., 0]) and np.all(deltas[..., 3] > deltas[..., 1]):
                # Похоже на формат x1, y1, x2, y2
                pred_boxes = deltas.copy()
            else:
                # Формат (x, y, w, h) - центр и размеры
                x = deltas[..., 0]
                y = deltas[..., 1]
                w = deltas[..., 2]
                h = deltas[..., 3]
                
                # Преобразуем в x1, y1, x2, y2
                pred_boxes = np.zeros_like(deltas)
                pred_boxes[..., 0] = x - 0.5 * w
                pred_boxes[..., 1] = y - 0.5 * h
                pred_boxes[..., 2] = x + 0.5 * w
                pred_boxes[..., 3] = y + 0.5 * h
        else:
            # Стандартная параметризация RetinaNet: смещения относительно якорей
            # В RetinaNet часто используется параметризация:
            # dx = (x_pred - x_anchor) / w_anchor
            # dy = (y_pred - y_anchor) / h_anchor
            # dw = log(w_pred / w_anchor)
            # dh = log(h_pred / h_anchor)
            
            dx = deltas[..., 0]
            dy = deltas[..., 1]
            dw = deltas[..., 2]
            dh = deltas[..., 3]
            
            # Избегаем деления на ноль
            widths_safe = np.maximum(widths, 1.0)
            heights_safe = np.maximum(heights, 1.0)
            
            # Применение смещений
            # Стандартная параметризация RetinaNet/Faster R-CNN
            pred_ctr_x = dx * widths_safe + ctr_x
            pred_ctr_y = dy * heights_safe + ctr_y
            
            # Ограничиваем экспоненту для стабильности
            dw_clipped = np.clip(dw, -10, 10)
            dh_clipped = np.clip(dh, -10, 10)
            
            pred_w = np.exp(dw_clipped) * widths_safe
            pred_h = np.exp(dh_clipped) * heights_safe
            
            # Обратно в x1, y1, x2, y2
            pred_boxes = np.zeros_like(deltas)
            pred_boxes[..., 0] = pred_ctr_x - 0.5 * pred_w
            pred_boxes[..., 1] = pred_ctr_y - 0.5 * pred_h
            pred_boxes[..., 2] = pred_ctr_x + 0.5 * pred_w
            pred_boxes[..., 3] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes
    
    def _filter_duplicate_boxes(self, boxes: np.ndarray, scores: np.ndarray, 
                                class_ids: np.ndarray, 
                                center_threshold: float = 0.08,
                                iou_threshold: float = 0.3) -> np.ndarray:
        """
        Фильтрация дубликатов по близости центров, IoU и aspect ratio.
        
        Удаляет боксы, которые:
        1. Имеют одинаковый класс
        2. Имеют высокий IoU (> iou_threshold) ИЛИ близкие центры с похожими размерами
        3. Имеют очень разные aspect ratios (> 2x разница) но обрамляют один объект
        
        Это помогает удалить дубликаты с разной ориентацией (вертикальный/горизонтальный),
        которые могут иметь низкий IoU из-за разной ориентации, но обрамляют один объект.
        
        Args:
            boxes: [N, 4] боксы в формате [x1, y1, x2, y2]
            scores: [N] уверенности
            class_ids: [N] идентификаторы классов
            center_threshold: Порог близости центров (относительно среднего размера бокса)
            iou_threshold: Порог IoU для определения дубликатов
            
        Returns:
            Индексы боксов для сохранения
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        # Вычисляем центры и размеры боксов
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # Вычисляем aspect ratios (соотношения сторон)
        aspect_ratios = widths / (heights + 1e-8)
        
        # Сортируем по уверенности (от большей к меньшей)
        indices = np.argsort(scores)[::-1]
        keep = []
        
        # Вычисляем средний размер боксов для нормализации расстояний
        avg_size = np.sqrt(np.mean(areas))
        if avg_size < 1.0:
            avg_size = 1.0
        
        for i in indices:
            current_idx = i
            is_duplicate = False
            
            # Проверяем со всеми уже сохраненными боксами
            for kept_idx in keep:
                # Только для одинаковых классов
                if class_ids[current_idx] != class_ids[kept_idx]:
                    continue
                
                # Вычисляем IoU
                iou = compute_iou(boxes[current_idx], boxes[kept_idx:kept_idx+1])[0]
                
                # Если IoU высокий - это явный дубликат
                if iou > iou_threshold:
                    is_duplicate = True
                    break
                
                # Расстояние между центрами
                center_dist = np.sqrt(
                    (centers_x[current_idx] - centers_x[kept_idx])**2 +
                    (centers_y[current_idx] - centers_y[kept_idx])**2
                )
                
                # Нормализованное расстояние
                normalized_dist = center_dist / avg_size
                
                # Если центры очень близки
                if normalized_dist < center_threshold:
                    # Проверяем соотношение размеров (площадей)
                    size_ratio = min(areas[current_idx], areas[kept_idx]) / max(areas[current_idx], areas[kept_idx])
                    
                    # Проверяем aspect ratio - если очень разные (> 2x разница),
                    # но центры близки и площади похожи - это может быть дубликат
                    aspect_ratio_diff = abs(aspect_ratios[current_idx] - aspect_ratios[kept_idx])
                    aspect_ratio_ratio = min(aspect_ratios[current_idx], aspect_ratios[kept_idx]) / \
                                        (max(aspect_ratios[current_idx], aspect_ratios[kept_idx]) + 1e-8)
                    
                    # Если размеры похожи (разница < 40%)
                    if size_ratio > 0.6:
                        # Проверка пересечения
                        x1_inter = max(boxes[current_idx][0], boxes[kept_idx][0])
                        y1_inter = max(boxes[current_idx][1], boxes[kept_idx][1])
                        x2_inter = min(boxes[current_idx][2], boxes[kept_idx][2])
                        y2_inter = min(boxes[current_idx][3], boxes[kept_idx][3])
                        
                        if x2_inter > x1_inter and y2_inter > y1_inter:
                            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                            min_area = min(areas[current_idx], areas[kept_idx])
                            
                            # Если пересечение > 30% от меньшего бокса - дубликат (более агрессивно)
                            if intersection / min_area > 0.3:
                                is_duplicate = True
                                break
                    
                    # Если aspect ratios очень разные (> 1.5x), но центры близки и площади похожи
                    # - это может быть дубликат с разной ориентацией
                    if aspect_ratio_diff > 0.5 and size_ratio > 0.5:
                        # Дополнительная проверка: если один бокс "внутри" другого
                        x1_inter = max(boxes[current_idx][0], boxes[kept_idx][0])
                        y1_inter = max(boxes[current_idx][1], boxes[kept_idx][1])
                        x2_inter = min(boxes[current_idx][2], boxes[kept_idx][2])
                        y2_inter = min(boxes[current_idx][3], boxes[kept_idx][3])
                        
                        if x2_inter > x1_inter and y2_inter > y1_inter:
                            intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                            min_area = min(areas[current_idx], areas[kept_idx])
                            
                            # Если пересечение > 25% от меньшего бокса - дубликат (более агрессивно)
                            if intersection / min_area > 0.25:
                                is_duplicate = True
                                break
                    
                    # Дополнительная проверка: если центры очень близки (в пределах 5% от среднего размера)
                    # и площади похожи, даже без большого пересечения - это может быть дубликат
                    if normalized_dist < 0.05 and size_ratio > 0.65:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                keep.append(current_idx)
        
        return np.array(keep, dtype=np.int32)

