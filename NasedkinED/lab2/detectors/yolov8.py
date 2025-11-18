# detectors/yolov8.py
import cv2
import numpy as np

from .base import ObjectDetector


class YOLOv8Detector(ObjectDetector):
    def __init__(self, model_path, conf_threshold=0.4, iou_threshold=0.45):
        super().__init__(conf_threshold, iou_threshold)
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # или DNN_TARGET_CUDA если есть GPU

        # Классы COCO (YOLOv8 использует те же)
        self.classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        self.target_classes = {"car", "bus", "truck", "motorcycle", "bicycle"}  # motobike → motorcycle

    def preprocess(self, image):
        # YOLOv8 ожидает 640x640
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        return blob

    def postprocess(self, image, outs):
        h, w = image.shape[:2]
        detections = []

        # YOLOv8 ONNX выводит тензор [1, 84, 8400] → транспонируем в [8400, 84]
        predictions = np.squeeze(outs[0]).T

        # Первые 4 — координаты бокса, потом 80 классов
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        # Центры и размеры → x1, y1, x2, y2
        box_centers = boxes[:, :2]
        box_sizes = boxes[:, 2:]
        x1 = box_centers[:, 0] - box_sizes[:, 0] / 2
        y1 = box_centers[:, 1] - box_sizes[:, 1] / 2
        x2 = box_centers[:, 0] + box_sizes[:, 0] / 2
        y2 = box_centers[:, 1] + box_sizes[:, 1] / 2

        # Масштабируем в размер изображения
        x1 *= w / 640
        y1 *= h / 640
        x2 *= w / 640
        y2 *= h / 640

        # Находим лучший класс и confidence
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Фильтруем по порогу и нужным классам
        mask = confidences > self.conf_threshold
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]

        indices = cv2.dnn.NMSBoxes(
            [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in range(len(x1))],
            confidences.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )

        # Универсальная обработка: работает и со старым, и с новым OpenCV
        if len(indices) > 0:
            if isinstance(indices[0], (list, tuple, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = indices.flatten()

            for i in indices:
                i = int(i)
                class_name = self.classes[class_ids[i]]
                if class_name not in self.target_classes:
                    continue
                detections.append({
                    'class_id': int(class_ids[i]),
                    'class_name': class_name,
                    'confidence': float(confidences[i]),
                    'box': (int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i]))
                })

        return detections
