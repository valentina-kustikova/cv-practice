import cv2
import numpy as np
from detector_base import ObjectDetector

class SSDMobileNet(ObjectDetector):
    def _load_network(self, model, config):
        return cv2.dnn.readNetFromTensorflow(model, config)

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

    def postprocess(self, image, outputs):
        h, w = image.shape[:2]
        results = []
        
        for detection in outputs[0][0, 0, :, :]:
            score = float(detection[2])
            if score > self.conf_thresh:
                class_id = int(detection[1]) - 1 
                if 0 <= class_id < len(self.classes):
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    results.append({
                        'class': self.classes[class_id],
                        'conf': score,
                        'box': (x1, y1, x2, y2)
                    })
        return results

class YOLOv4(ObjectDetector):
    def _load_network(self, model, config):
        net = cv2.dnn.readNetFromDarknet(config, model)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    def postprocess(self, image, outputs):
        h, w = image.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_thresh:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                results.append({
                    'class': self.classes[class_ids[i]],
                    'conf': confidences[i],
                    'box': (x, y, x + w_box, y + h_box)
                })
        return results

class FasterRCNN(ObjectDetector):
    def _load_network(self, model, config):
        return cv2.dnn.readNetFromTensorflow(model, config)

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, swapRB=True, crop=False)

    def postprocess(self, image, outputs):
        h, w = image.shape[:2]
        results = []
        
        # Выход аналогичен SSD: [1, 1, N, 7]
        for detection in outputs[0][0, 0, :, :]:
            score = float(detection[2])
            if score > self.conf_thresh:
                class_id = int(detection[1])
                
                if class_id >= len(self.classes) or class_id < 0:
                    continue

                x1 = int(detection[3] * w)
                y1 = int(detection[4] * h)
                x2 = int(detection[5] * w)
                y2 = int(detection[6] * h)

                results.append({
                    'class': self.classes[class_id],
                    'conf': score,
                    'box': (x1, y1, x2, y2)
                })
        return results