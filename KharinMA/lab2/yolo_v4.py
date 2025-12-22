from pathlib import Path

import cv2
import numpy as np

from base_detector import BaseDetector, Detection


class YoloV4CocoDetector(BaseDetector, model_name="yolo_v4_coco"):
    def __init__(
        self,
        model_path,
        config_path,
        classes_path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.classes_path = Path(classes_path)

        class_names = self._load_classes(self.classes_path)

        super().__init__(
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            class_names=class_names,
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Не найден файл модели: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Не найден файл конфигурации: {self.config_path}")

        self.net = cv2.dnn.readNetFromDarknet(
            str(self.config_path),
            str(self.model_path),
        )
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()

        self.input_size = (608, 608)

    @staticmethod
    def _load_classes(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _preprocess(self, image):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )

        meta = {"orig_size": (w, h)}
        return blob, meta

    def _forward(self, blob):
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)
        return outputs

    def _postprocess(self, image, outputs, meta=None):
        h, w = image.shape[:2]

        boxes_xywh = []
        scores = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores_det = detection[5:]
                class_id = int(np.argmax(scores_det))
                confidence = float(scores_det[class_id])

                if confidence < self.conf_threshold:
                    continue

                cx, cy, bw, bh = detection[0:4]
                cx *= w
                cy *= h
                bw *= w
                bh *= h

                x_min = int(cx - bw / 2)
                y_min = int(cy - bh / 2)
                x_max = int(cx + bw / 2)
                y_max = int(cy + bh / 2)

                x_min = max(0, min(x_min, w - 1))
                x_max = max(0, min(x_max, w - 1))
                y_min = max(0, min(y_min, h - 1))
                y_max = max(0, min(y_max, h - 1))
                if x_max <= x_min or y_max <= y_min:
                    continue

                bw_int = x_max - x_min
                bh_int = y_max - y_min

                boxes_xywh.append([x_min, y_min, bw_int, bh_int])
                scores.append(confidence)
                class_ids.append(class_id)

        if not boxes_xywh:
            return []

        detections = []

        unique_classes = set(class_ids)
        for c in unique_classes:
            idxs_c = [i for i, cid in enumerate(class_ids) if cid == c]
            boxes_c = [boxes_xywh[i] for i in idxs_c]
            scores_c = [scores[i] for i in idxs_c]
            # Это список номеров (индексов) тех рамок из исходного списка boxes_c, которые выжили после чистки.
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_c,
                scores=scores_c,
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
            )
            if len(indices) == 0:
                continue

            for k in np.array(indices).flatten():
                i = idxs_c[k]
                x, y, bw_int, bh_int = boxes_xywh[i]
                x_min = x
                y_min = y
                x_max = x + bw_int
                y_max = y + bh_int
                class_id = class_ids[i]
                confidence = scores[i]

                class_name = (
                    self.class_names[class_id]
                    if 0 <= class_id < len(self.class_names)
                    else str(class_id)
                )

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )
                )

        return detections