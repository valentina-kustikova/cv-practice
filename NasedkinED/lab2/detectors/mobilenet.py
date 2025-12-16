import cv2

from .base import ObjectDetector


class MobileNetSSDDetector(ObjectDetector):
    def __init__(self, model_path, config_path, conf_threshold=0.2):
        super().__init__(conf_threshold)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.target_classes = {"car", "bus", "motorbike", "bicycle", "train"}

        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    def postprocess(self, image, outs):
        h, w = image.shape[:2]
        detections = []

        # outs — это список с одним элементом, берём его
        out = outs[0]  # shape: (1, 1, N, 7)

        # Правильно: out[0, 0, :, :]
        out = out[0, 0]  # теперь shape: (N, 7)

        for i in range(out.shape[0]):
            confidence = out[i, 2]

            # Проверяем, что confidence — скаляр и больше порога
            if confidence > self.conf_threshold:
                class_id = int(out[i, 1])
                if class_id >= len(self.classes):
                    continue
                class_name = self.classes[class_id]

                if class_name not in self.target_classes:
                    continue

                x1 = int(out[i, 3] * w)
                y1 = int(out[i, 4] * h)
                x2 = int(out[i, 5] * w)
                y2 = int(out[i, 6] * h)

                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'box': (x1, y1, x2 - x1, y2 - y1)
                })

        return detections
