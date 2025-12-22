from pathlib import Path

import cv2

from base_detector import BaseDetector, Detection


class SsdMobilenetV2CocoDetector(BaseDetector, model_name="ssd_mobilenet_v2_coco"):
    def __init__(
        self,
        model_path,
        config_path,
        classes_path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
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

        self.net = cv2.dnn.readNetFromTensorflow(
            str(self.model_path),
            str(self.config_path),
        )

        self.input_size = (300, 300)

    @staticmethod
    def _load_classes(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _preprocess(self, image):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )

        meta = {"orig_size": (w, h)}
        return blob, meta

    def _forward(self, blob):
        self.net.setInput(blob)
        outputs = self.net.forward()
        return outputs

    def _postprocess(self, image, outputs, meta=None):
        h, w = image.shape[:2]
        print(f"Размер выхода SSD: {outputs.shape}")
        boxes_xywh = []
        scores = []
        class_ids = []

        out = outputs[0, 0, :, :]

        for det in out:
            score = float(det[2])
            if score < self.conf_threshold:
                continue

            class_id_tf = int(det[1])
            class_idx = class_id_tf - 1

            if not (0 <= class_idx < len(self.class_names)):
                continue

            x_min = int(det[3] * w)
            y_min = int(det[4] * h)
            x_max = int(det[5] * w)
            y_max = int(det[6] * h)

            x_min = max(0, min(x_min, w - 1))
            x_max = max(0, min(x_max, w - 1))
            y_min = max(0, min(y_min, h - 1))
            y_max = max(0, min(y_max, h - 1))
            if x_max <= x_min or y_max <= y_min:
                continue

            bw = x_max - x_min
            bh = y_max - y_min

            boxes_xywh.append([x_min, y_min, bw, bh])
            scores.append(score)
            class_ids.append(class_idx)

        if not boxes_xywh:
            return []

        detections = []

        unique_classes = set(class_ids)
        for c in unique_classes:
            idxs_c = [i for i, cid in enumerate(class_ids) if cid == c]
            boxes_c = [boxes_xywh[i] for i in idxs_c]
            scores_c = [scores[i] for i in idxs_c]

            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_c,
                scores=scores_c,
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
            )
            if len(indices) == 0:
                continue

            for k in indices:
                i = idxs_c[k]
                x, y, bw, bh = boxes_xywh[i]
                x_min = x
                y_min = y
                x_max = x + bw
                y_max = y + bh
                class_idx = class_ids[i]
                score = scores[i]

                detections.append(
                    Detection(
                        class_id=class_idx,
                        class_name=self.class_names[class_idx],
                        confidence=score,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )
                )

        return detections