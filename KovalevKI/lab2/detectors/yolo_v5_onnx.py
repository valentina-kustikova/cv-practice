import cv2
import numpy as np
from .base import ObjectDetector

class YOLOv5sOpenCVDetector(ObjectDetector):
    def _load_model(self, model_path, config_path=None):
        return cv2.dnn.readNetFromONNX(model_path)

    def preprocess(self, image):
        input_size = 640
        h, w = image.shape[:2]
        scale = min(input_size / w, input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(padded, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        return blob, scale, (new_w, new_h)

    def postprocess(self, outputs, image_shape, scale, padded_size):
        h_orig, w_orig = image_shape[:2]
        preds = outputs[0]  # (1, 25200, 85) → (25200, 85)
        if preds.ndim == 3:
            preds = preds[0]

        boxes, confidences, class_ids = [], [], []

        for det in preds:
            xc, yc, w, h = det[:4]
            conf = det[4]
            if conf <= self.conf_threshold:
                continue
            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            confidence = float(conf * class_scores[class_id])
            if confidence <= self.conf_threshold:
                continue

            pad_w = (640 - padded_size[0]) / 2
            pad_h = (640 - padded_size[1]) / 2
            xc_adj = xc / scale
            yc_adj = yc / scale
            w_adj = w / scale
            h_adj = h / scale

            x1 = int(xc_adj - w_adj / 2)
            y1 = int(yc_adj - h_adj / 2)
            boxes.append([x1, y1, int(w_adj), int(h_adj)])
            confidences.append(confidence)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return [], [], []
        indices = indices.flatten()
        return (
            [boxes[i] for i in indices],
            [confidences[i] for i in indices],
            [class_ids[i] for i in indices]
        )

    def detect(self, image):
        blob, scale, padded_size = self.preprocess(image)
        self.model.setInput(blob)
        outputs = self.model.forward()
        return self.postprocess(outputs, image.shape, scale, padded_size)