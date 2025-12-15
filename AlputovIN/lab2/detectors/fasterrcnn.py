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

        # Распаковка архива (код инициализации без изменений)
        if not os.path.exists(model_path):
            tar_path = os.path.join(models_dir, 'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz')
            if os.path.exists(tar_path):
                print("Распаковка архива Faster R-CNN...")
                try:
                    import shutil
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extract('faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', models_dir)
                        extracted_path = os.path.join(models_dir, 'faster_rcnn_inception_v2_coco_2018_01_28', 'frozen_inference_graph.pb')
                        if os.path.exists(extracted_path):
                            shutil.move(extracted_path, model_path)
                            try:
                                os.rmdir(os.path.join(models_dir, 'faster_rcnn_inception_v2_coco_2018_01_28'))
                            except OSError: pass
                except Exception as e:
                    raise FileNotFoundError(f"Не удалось распаковать архив: {e}")

        self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Понижаем порог для повышения TPR (старые модели могут быть менее уверены)
        self.conf_th = 0.3

        # Список классов COCO (стандартный для TF моделей)
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

        # Классы, которые мы хотим детектировать
        self.vehicle_classes = {'car', 'bus', 'truck', 'motorcycle'}

        # Создаем set ID классов для быстрой проверки
        self._vehicle_class_ids = set()
        for i, name in enumerate(self.class_names):
            if name in self.vehicle_classes:
                self._vehicle_class_ids.add(i)

    def detect(self, image):
        """
        Детектирует транспортные средства.
        Использует упрощенный подход: resize до 600x600 и прямое масштабирование координат.
        """
        if image is None or image.size == 0:
            return []

        # Сохраняем исходные размеры
        img_height, img_width = image.shape[:2]

        # Предобработка: просто сжимаем картинку. OpenCV сам разберется с нормализацией для TF моделей.
        # size=(600, 600) стандарт для этой архитектуры
        blob = cv2.dnn.blobFromImage(image, size=(600, 600), swapRB=True, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward()

        # Валидация вывода (исправление ошибки ValueError)
        if outputs is None or outputs.size == 0:
            return []

        detections = []

        # Проходим по всем детекциям (цикл как в примере)
        # outputs shape: [1, 1, N, 7]
        for detection in outputs[0, 0]:
            confidence = float(detection[2])
            class_id = int(detection[1])

            # 1. Проверяем класс (входит ли в список нужных)
            # 2. Проверяем порог уверенности
            if class_id in self._vehicle_class_ids and confidence > self.conf_th:
                # Координаты возвращаются нормализованными (0 to 1)
                # Умножаем их на исходные размеры картинки (как в примере)
                x1 = int(detection[3] * img_width)
                y1 = int(detection[4] * img_height)
                x2 = int(detection[5] * img_width)
                y2 = int(detection[6] * img_height)

                # Клиппинг координат (чтобы не вылезти за пределы)
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))

                # Проверка на валидность бокса (ширина и высота > 0)
                if x2 > x1 and y2 > y1:
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)

                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })

        return detections