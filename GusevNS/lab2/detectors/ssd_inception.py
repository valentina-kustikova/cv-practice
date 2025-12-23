from pathlib import Path
import cv2
import numpy as np
import tarfile
import io
import urllib.request
from .base import BaseDetector, DetectionResult


class SsdInceptionDetector(BaseDetector):
    def __init__(self, model_dir):
        super().__init__(
            model_dir=model_dir,
            score_threshold=0.2,
            nms_threshold=0.4,
            class_filter=["car", "truck", "bus", "motorcycle"],
        )

    def load_class_names(self):
        names_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/object_detection_classes_coco.txt"
        names_path = self.model_dir / "coco.names"
        self.download_file(names_url, names_path)
        with open(names_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    def ensure_model_files(self):
        tar_url = "http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz"
        pbtxt_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_inception_v2_coco_2017_11_17.pbtxt"
        self.pb_path = self.model_dir / "ssd_inception_v2_coco.pb"
        self.pbtxt_path = self.model_dir / "ssd_inception_v2_coco.pbtxt"
        if not self.pb_path.exists():
            print(f"Downloading SSD Inception V2 weights from {tar_url}...")
            req = urllib.request.Request(tar_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                tar_data = io.BytesIO(response.read())
            with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("frozen_inference_graph.pb"):
                        member.name = self.pb_path.name
                        tar.extract(member, self.model_dir)
                        break
        self.download_file(pbtxt_url, self.pbtxt_path)

    def load_network(self):
        self.net = cv2.dnn_DetectionModel(str(self.pb_path), str(self.pbtxt_path))
        self.net.setInputSize(300, 300)
        self.net.setInputScale(1.0)
        self.net.setInputMean((0, 0, 0))
        self.net.setInputSwapRB(True)

    def preprocess(self, image):
        return image

    def infer(self, blob):
        return self.net.detect(blob, confThreshold=self.score_threshold, nmsThreshold=self.nms_threshold)

    def postprocess(self, outputs, original_shape):
        classes, confidences, boxes = outputs
        results = []
        if classes is None or len(classes) == 0:
            return results
        classes = np.array(classes).flatten()
        confidences = np.array(confidences).flatten()
        for class_id, confidence, box in zip(classes, confidences, boxes):
            adjusted_id = int(class_id) - 1
            if adjusted_id < 0 or adjusted_id >= len(self.class_names):
                continue
            label = self.class_names[adjusted_id]
            if label not in self.class_filter:
                continue
            x, y, w, h = box
            results.append(DetectionResult(adjusted_id, label, float(confidence), (int(x), int(y), int(w), int(h))))
        return results

    def detect(self, image):
        outputs = self.infer(image)
        return self.postprocess(outputs, image.shape[:2])

