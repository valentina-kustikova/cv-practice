from .base import Detector
import cv2
import numpy as np
import os
import tarfile

class FasterRCNNDetector(Detector):
    """
    Детектор на основе Faster R-CNN (TensorFlow формат).
    """

    def __init__(self):
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(models_dir, 'frozen_inference_graph.pb')
        config_path = os.path.join(models_dir, 'faster_rcnn_inception_v2_coco.pbtxt')

        # Распаковываем архив, если файл не существует
        if not os.path.exists(model_path):
            tar_path = os.path.join(models_dir, 'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz')
            if os.path.exists(tar_path):
                print("Распаковка архива Faster R-CNN...")
                try:
                    import shutil
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        # Извлекаем только нужный файл
                        tar.extract('faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', models_dir)
                        # Перемещаем файл в нужное место
                        extracted_path = os.path.join(models_dir, 'faster_rcnn_inception_v2_coco_2018_01_28', 'frozen_inference_graph.pb')
                        if os.path.exists(extracted_path):
                            shutil.move(extracted_path, model_path)
                            # Удаляем пустую папку
                            try:
                                os.rmdir(os.path.join(models_dir, 'faster_rcnn_inception_v2_coco_2018_01_28'))
                            except OSError:
                                pass  # Папка не пустая или уже удалена
                except Exception as e:
                    raise FileNotFoundError(f"Не удалось распаковать архив Faster R-CNN: {e}")
            else:
                raise FileNotFoundError(f"Файл модели не найден: {model_path}. Также не найден архив: {tar_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        # Загружаем модель с оптимизацией для CPU
        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_th = 0.5
        # Кэш для валидных индексов классов транспорта (оптимизация)
        self._vehicle_class_ids = None
        self.class_names = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
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
        # Предвычисляем индексы классов транспорта для быстрой фильтрации
        self._vehicle_class_ids = {i for i, name in enumerate(self.class_names)
                                   if name in self.vehicle_classes}

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
        blob = cv2.dnn.blobFromImage(image, size=(600, 600), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Валидация вывода модели
        if outputs is None or outputs.size == 0:
            return []

        # Формат вывода Faster R-CNN: [1, 1, N, 7]
        # где 7 = [batch_id, class_id, confidence, x1, y1, x2, y2]
        # Координаты нормализованы [0, 1]
        detections = outputs[0, 0]

        if detections.size == 0:
            return []

        # Векторизованная фильтрация (оптимизация)
        confidences = detections[:, 2].astype(np.float32)
        class_ids = detections[:, 1].astype(np.int32)

        # Фильтруем по confidence и классам транспорта одновременно
        valid_mask = (confidences > self.conf_th) & np.isin(class_ids, list(self._vehicle_class_ids))

        if not np.any(valid_mask):
            return []

        # Применяем маску
        valid_detections = detections[valid_mask]
        valid_confidences = confidences[valid_mask]
        valid_class_ids = class_ids[valid_mask]

        # Векторизованное преобразование координат (оптимизация)
        coords = valid_detections[:, 3:7]  # [N, 4] - x1, y1, x2, y2 нормализованные
        x1 = (coords[:, 0] * w).astype(np.int32)
        y1 = (coords[:, 1] * h).astype(np.int32)
        x2 = (coords[:, 2] * w).astype(np.int32)
        y2 = (coords[:, 3] * h).astype(np.int32)

        # Векторизованный клиппинг
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        # Фильтруем валидные bbox
        valid_bbox_mask = (x2 > x1) & (y2 > y1) & (x2 - x1 > 2) & (y2 - y1 > 2)

        if not np.any(valid_bbox_mask):
            return []

        # Применяем маску
        x1 = x1[valid_bbox_mask]
        y1 = y1[valid_bbox_mask]
        x2 = x2[valid_bbox_mask]
        y2 = y2[valid_bbox_mask]
        valid_confidences = valid_confidences[valid_bbox_mask]
        valid_class_ids = valid_class_ids[valid_bbox_mask]

        # Формируем результаты
        results = []
        for i in range(len(x1)):
            class_id = int(valid_class_ids[i])
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            results.append({
                'class': class_name,
                'confidence': float(valid_confidences[i]),
                'bbox': [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
            })

        return results
