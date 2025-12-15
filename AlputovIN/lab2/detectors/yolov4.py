from .base import Detector
import cv2
import numpy as np
import os

class YOLOv4Detector(Detector):
    def __init__(self):
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        weights_path = os.path.join(models_dir, 'yolov4.weights')
        config_path = os.path.join(models_dir, 'yolov4.cfg')

        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"Файлы YOLOv4 не найдены в {models_dir}. Запустите download_models.py")

        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Универсальный способ получения имен выходных слоев для разных версий OpenCV
        try:
            self.output_layers = self.net.getUnconnectedOutLayersNames()
        except:
            # Fallback для старых версий
            layers = self.net.getLayerNames()
            self.output_layers = [layers[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.input_size = (608, 608)
        self.conf_th = 0.3  # Чуть снизил порог
        self.nms_th = 0.4

        # Стандартные COCO классы
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

        # То, что нас интересует
        self.vehicle_classes = {'car', 'bus', 'truck', 'motorcycle'}

    def detect(self, image):
        if image is None or image.size == 0:
            return []

        H, W = image.shape[:2]

        # Blob
        blob = cv2.dnn.blobFromImage(image, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                # detection: [center_x, center_y, w, h, confidence, class_scores...]
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > self.conf_th:
                    class_name = self.class_names[class_id]

                    if class_name in self.vehicle_classes:
                        # YOLO возвращает относительные координаты центра и размера
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # Перевод в верхний левый угол
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(confidence)
                        class_ids.append(class_id)

        results = []
        if len(boxes) > 0:
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_th, self.nms_th)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y, w, h = boxes[i]
                    # Коррекция выхода за границы
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(W, x + w)
                    y2 = min(H, y + h)

                    results.append({
                        'class': self.class_names[class_ids[i]],
                        'confidence': confidences[i],
                        'bbox': [x, y, x2, y2]
                    })

        return results
