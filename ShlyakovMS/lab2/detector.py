from abc import ABC, abstractmethod
import cv2
import os
import urllib.request
import numpy as np

# Папка для автоматического кэширования моделей
CACHE_DIR = os.path.join(os.path.dirname(__file__), "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Скачиваю {os.path.basename(dest_path)} ...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Готово: {os.path.basename(dest_path)}")
    else:
        print(f"Уже есть: {os.path.basename(dest_path)}")

class BaseDetector(ABC):
    def __init__(self, weights_url, config_url=None, classes_url=None,
                 input_size=(416, 416), scale=1/255.0, mean=(0,0,0), swap_rb=True,
                 framework="darknet"):

        weights_path = os.path.join(CACHE_DIR, os.path.basename(weights_url.split("?")[0]))
        config_path = None
        classes_path = os.path.join(CACHE_DIR, "coco.names")

        download_file(weights_url, weights_path)
        if config_url:
            config_path = os.path.join(CACHE_DIR, os.path.basename(config_url.split("?")[0]))
            download_file(config_url, config_path)
        if classes_url:
            classes_path = os.path.join(CACHE_DIR, os.path.basename(classes_url.split("?")[0]))
            download_file(classes_url, classes_path)

        if framework == "darknet" and config_path:
            self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        elif framework == "caffe" and config_path:
            self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
        else:
            self.net = cv2.dnn.readNet(weights_path)

        self.input_size = input_size
        self.scale = scale
        self.mean = mean
        self.swap_rb = swap_rb

        with open(classes_path, 'r', encoding='utf-8') as f:
            self.classes = [line.strip().lower() for line in f.readlines()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(int)
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def postprocess(self, image, outputs, conf_threshold=0.5, nms_threshold=0.4):
        pass

    def detect(self, image, conf_threshold=0.5, nms_threshold=0.4):
        blob = self.preprocess(image)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)
        return self.postprocess(image, outputs, conf_threshold, nms_threshold)


class YOLOv3Detector(BaseDetector):
    def __init__(self):
        super().__init__(
            weights_url="https://pjreddie.com/media/files/yolov3.weights",
            config_url="https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            classes_url="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            input_size=(416, 416),
            scale=1/255.0,
            mean=(0,0,0),
            swap_rb=True,
            framework="darknet"
        )

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, self.scale, self.input_size,
                                     self.mean, swapRB=self.swap_rb, crop=False)

    def postprocess(self, image, outputs, conf_threshold=0.5, nms_threshold=0.4):
        h, w = image.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and self.classes[class_id] in self.vehicle_classes:
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    x = int(cx - bw / 2)
                    y = int(cy - bh / 2)
                    boxes.append([x, y, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                det = {
                    'box': box,
                    'confidence': confidences[i],
                    'class_name': self.classes[class_ids[i]],
                    'class_id': class_ids[i]
                }
                detections.append(det)
        return detections


class YOLOv4TinyDetector(BaseDetector):
    def __init__(self):
        super().__init__(
            weights_url="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
            config_url="https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
            classes_url="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            input_size=(416, 416),
            scale=1/255.0,
            mean=(0,0,0),
            swap_rb=True,
            framework="darknet"
        )

    preprocess = YOLOv3Detector.preprocess
    postprocess = YOLOv3Detector.postprocess


class MobileNetSSDDetector(BaseDetector):
    def __init__(self):
        weights_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
        config_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        
        weights_path = os.path.join(CACHE_DIR, "mobilenet_iter_73000.caffemodel")
        config_path = os.path.join(CACHE_DIR, "deploy.prototxt")
        
        download_file(weights_url, weights_path)
        download_file(config_url, config_path)
        
        self.net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
        
        self.input_size = (300, 300)
        self.scale = 0.007843
        self.mean = (127.5, 127.5, 127.5)
        self.swap_rb = False
        
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype(int)
        self.vehicle_classes = ['car', 'bus', 'bicycle', 'motorbike']

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, self.scale, self.input_size,
                                     self.mean, swapRB=self.swap_rb, crop=False)

    def postprocess(self, image, outputs, conf_threshold=0.5, nms_threshold=0.4):
        h, w = image.shape[:2]
        detections = []
    
        detections_raw = outputs[0][0, 0, :, :]
    
        for detection in detections_raw:
            confidence = float(detection[2])
            class_id = int(detection[1])
    
            if confidence > conf_threshold and 0 < class_id < len(self.classes):
                class_name = self.classes[class_id]
                if class_name in self.vehicle_classes:
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)
    
                    # clamp
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
    
                    detections.append({
                        'box': [x1, y1, x2 - x1, y2 - y1],
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    })

        if detections:
            boxes = [d['box'] for d in detections]
            confs = [d['confidence'] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, nms_threshold)
            if len(indices) > 0:
                detections = [detections[i] for i in indices.flatten()]
    
        return detections