import cv2
import numpy as np
from .base_detector import BaseDetector


class SSDDetector(BaseDetector):
    def __init__(self, model_path, config_path, classes_path, conf_threshold=0.3):
        self.classes_path = classes_path
        self.load_classes()
        super().__init__(model_path, config_path, conf_threshold)

    def load_classes(self):
        # SSD MobileNet использует 20 классов VOC (не COCO!)
        self.voc_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
        print(f"SSD using {len(self.voc_classes)} VOC classes")

        # Прямой маппинг VOC ID -> COCO имена
        self.class_id_to_coco_name = {
            7: 'car',  # car -> car
            6: 'bus',  # bus -> bus
            14: 'motorcycle',  # motorbike -> motorcycle
            2: 'bicycle',  # bicycle -> bicycle
            1: 'airplane',  # aeroplane -> airplane
            19: 'train'  # train -> train
        }

    def load_model(self):
        print(f"Loading SSD model from: {self.model_path}")
        print(f"Config from: {self.config_path}")

        self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
        if self.net.empty():
            raise Exception("Failed to load SSD model")

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("SSD model loaded successfully")

    def preprocess(self, image):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        return blob

    def postprocess(self, outputs, image_shape):
        height, width = image_shape[:2]
        results = []

        if len(outputs) > 0:
            detection_output = outputs[0]  # [1, 1, 100, 7]

            for i in range(detection_output.shape[2]):
                detection = detection_output[0, 0, i]
                confidence = detection[2]
                class_id = int(detection[1])

                # Пропускаем background (class_id=0) и фильтруем по confidence
                if class_id > 0 and confidence > self.conf_threshold:
                    # Координаты в [0,1] диапазоне
                    x1 = int(detection[3] * width)
                    y1 = int(detection[4] * height)
                    x2 = int(detection[5] * width)
                    y2 = int(detection[6] * height)

                    # Проверяем валидность координат
                    if (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and
                            x2 <= width and y2 <= height and (x2 - x1) > 10):

                        # ПРЯМОЕ ПРИСВОЕНИЕ COCO ИМЕНИ
                        if class_id in self.class_id_to_coco_name:
                            class_name = self.class_id_to_coco_name[class_id]
                            voc_name = self.voc_classes[class_id] if class_id < len(
                                self.voc_classes) else f"class_{class_id}"
                        else:
                            class_name = f"class_{class_id}"
                            voc_name = class_name

                        results.append({
                            'class_id': class_id,
                            'class_name': class_name,  # COCO имя
                            'confidence': float(confidence),
                            'bbox': [x1, y1, x2, y2]
                        })

                        if len(results) <= 3:
                            print(f"SSD detection: {voc_name}({class_id}) -> {class_name} - {confidence:.3f}")

        print(f"SSD: {len(results)} detections")
        return results

    def get_output_layers(self):
        return ['detection_out']
