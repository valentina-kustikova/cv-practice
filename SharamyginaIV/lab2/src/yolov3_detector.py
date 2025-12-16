import cv2
import numpy as np
from .detector_base import ObjectDetector

class YOLOv3Detector(ObjectDetector):
    def __init__(self, model_path, config_path, classes_path, conf_threshold=0.5, nms_threshold=0.4):
        super().__init__(model_path, config_path, classes_path, conf_threshold, nms_threshold)
        self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def preprocess(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        return blob

    def postprocess(self, outputs, original_image_shape):
        """
        Постобработка выхода YOLOv3.
        Фильтрация, NMS, форматирование результатов.
        """
        height, width, channels = original_image_shape
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        final_boxes = []
        final_class_ids = []
        final_confidences = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_class_ids.append(class_ids[i])
                final_confidences.append(confidences[i])

        final_boxes_xyxy = []
        for box in final_boxes:
            x, y, w, h = box
            final_boxes_xyxy.append([x, y, x + w, y + h])

        return final_boxes_xyxy, final_class_ids, final_confidences

    def get_output_layers(self):
        return self.output_layers
