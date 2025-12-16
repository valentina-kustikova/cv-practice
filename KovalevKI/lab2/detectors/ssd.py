import cv2
import numpy as np
from .base import ObjectDetector

class MobileNetSSDDetector(ObjectDetector):
    def _load_model(self, model_path, config_path):
        return cv2.dnn.readNetFromCaffe(config_path, model_path)

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )

    def postprocess(self, outputs, image_shape):
        h, w = image_shape[:2]
        boxes, confidences, class_ids = [], [], []
        detections = outputs[0, 0, :, :]
        for i in range(detections.shape[0]):
            confidence = detections[i, 2]
            if confidence > self.conf_threshold:
                class_id = int(detections[i, 1])
                box = detections[i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return [], [], []
        idxs = indices.flatten()
        return [boxes[i] for i in idxs], [confidences[i] for i in idxs], [class_ids[i] for i in idxs]

    def detect(self, image):
        blob = self.preprocess(image)
        self.model.setInput(blob)
        outputs = self.model.forward()
        return self.postprocess(outputs, image.shape)