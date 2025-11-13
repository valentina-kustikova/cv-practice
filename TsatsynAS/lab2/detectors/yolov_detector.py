import cv2
import numpy as np
from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    def __init__(self, model_path, config_path, classes_path, conf_threshold=0.5, nms_threshold=0.4):
        self.classes_path = classes_path
        # Сначала вызываем родительский конструктор, потом загружаем классы
        super().__init__(model_path, config_path, conf_threshold, nms_threshold)
        self.load_classes()  # Теперь после super()

    def load_classes(self):
        print(f"Loading classes from: {self.classes_path}")
        try:
            with open(self.classes_path, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f.readlines() if line.strip()]
            print(f"✅ Successfully loaded {len(self.classes)} classes")

            # Проверим содержимое
            print("First 10 classes:", self.classes[:10])

        except Exception as e:
            print(f"❌ Error loading classes: {e}")
            print("Using fallback COCO classes")
            # Fallback COCO classes
            self.classes = [
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
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush'
            ]

        print(f"Total classes available: {len(self.classes)}")

    def load_model(self):
        print(f"Loading YOLO model from: {self.model_path}")
        self.net = cv2.dnn.readNet(self.model_path, self.config_path)
        if self.net.empty():
            raise Exception("Failed to load YOLO model")

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("YOLO model loaded successfully")

    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        return blob

    def postprocess(self, outputs, image_shape):
        height, width = image_shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        print(f"Available classes: {len(self.classes)}")

        # Временно уберем фильтрацию по классам для отладки
        print("DEBUG: Processing detections without class filtering...")

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # ДЕТАЛЬНАЯ ОТЛАДКА
                if confidence > 0.1:  # Временный низкий порог для отладки
                    print(
                        f"DEBUG: class_id={class_id}, confidence={confidence:.3f}, classes_available={len(self.classes)}")

                # ВРЕМЕННО: принимаем все классы с высоким confidence
                if confidence > self.conf_threshold:
                    center_x = detection[0] * width
                    center_y = detection[1] * height
                    w = detection[2] * width
                    h = detection[3] * height

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if w > 0 and h > 0 and x < width and y < height:
                        boxes.append([x, y, int(w), int(h)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                        # Логируем детекцию
                        if class_id < len(self.classes):
                            class_name = self.classes[class_id]
                        else:
                            class_name = f"unknown_{class_id}"
                        print(f"Detection: {class_name} (id:{class_id}) - {confidence:.3f}")

        print(f"Found {len(boxes)} detections before NMS")

        # NMS
        indices = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            if indices is not None:
                print(f"After NMS: {len(indices)} detections")
            else:
                indices = []

        results = []
        if len(indices) > 0:
            if hasattr(indices, 'flatten'):
                indices = indices.flatten()
            else:
                indices = [i[0] for i in indices]

            for idx in indices:
                if idx < len(boxes):
                    x, y, w, h = boxes[idx]
                    class_id = class_ids[idx]
                    confidence = confidences[idx]

                    if class_id < len(self.classes):
                        class_name = self.classes[class_id]
                    else:
                        class_name = f"unknown_{class_id}"

                    results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x, y, x + w, y + h]
                    })

        return results

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = self.net.getUnconnectedOutLayers()

        if hasattr(output_layers, 'ndim') and output_layers.ndim == 1:
            return [layer_names[i - 1] for i in output_layers]
        else:
            return [layer_names[i[0] - 1] for i in output_layers]
