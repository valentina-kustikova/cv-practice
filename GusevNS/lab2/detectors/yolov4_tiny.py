from pathlib import Path
import cv2
import numpy as np
from .base import BaseDetector, DetectionResult


class YoloV4TinyDetector(BaseDetector):
    def __init__(self, model_dir):
        self.input_size = (416, 416)
        super().__init__(
            model_dir=model_dir,
            score_threshold=0.25,
            nms_threshold=0.4,
            class_filter=["car", "truck", "bus", "motorbike"],
        )

    def load_class_names(self):
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        names_path = self.model_dir / "coco.names"
        self.download_file(names_url, names_path)
        with open(names_path, "r", encoding="utf-8") as source:
            return [line.strip() for line in source.readlines() if line.strip()]

    def ensure_model_files(self):
        cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
        self.cfg_path = self.download_file(cfg_url, self.model_dir / "yolov4-tiny.cfg")
        self.weights_path = self.download_file(weights_url, self.model_dir / "yolov4-tiny.weights")

    def load_network(self):
        self.net = cv2.dnn.readNetFromDarknet(str(self.cfg_path), str(self.weights_path))
        self.output_names = self.net.getUnconnectedOutLayersNames()

    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self.input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        return blob

    def infer(self, blob):
        return self.net.forward(self.output_names)

    def postprocess(self, outputs, original_shape):
        h, w = original_shape
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.score_threshold:
                    continue
                if not self.filter_by_vehicle(class_id):
                    continue
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score_threshold, self.nms_threshold)
        results = []
        if len(indices) == 0:
            return results
        for idx in indices.flatten():
            x, y, width, height = boxes[idx]
            x, y, width, height = self.scale_box((x, y, width, height), original_shape)
            label = self.get_label(class_ids[idx])
            results.append(DetectionResult(class_ids[idx], label, confidences[idx], (x, y, width, height)))
        return results

