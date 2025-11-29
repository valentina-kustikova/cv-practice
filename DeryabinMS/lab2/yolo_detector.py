from typing import List, Tuple
import numpy as np
import cv2 as cv

from detector_base import BaseDetector, Detection, DetectorConfig


class YOLOv4Detector(BaseDetector):

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        layer_names = self.net.getLayerNames()
        self.output_layer_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def _forward(self, blob):
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layer_names)
        return np.vstack(layer_outputs)

    def _postprocess(self, outputs: np.ndarray, image_shape: Tuple[int, int]) -> List[Detection]:
        img_h, img_w = image_shape
        detections: List[Detection] = []

        for row in outputs:
            object_score = float(row[4])
            if object_score < 0.1:
                continue

            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])

            confidence = object_score * class_score
            if confidence < self.config.conf_threshold:
                continue

            cx, cy, w, h = row[0], row[1], row[2], row[3]
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            if 0 <= class_id < len(self.config.classes):
                class_name = self.config.classes[class_id]
            else:
                class_name = f"class_{class_id}"

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                )
            )

        return detections
