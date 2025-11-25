from lib.nanodet import NanoDet
from lib.yolox import YoloX
from lib.ssd import SSD


model_urls = {
    'nanodet': 'https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx',
    'yolox': 'https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx',
    'ssd': 'https://github.com/WladislawFilatew/ssd/raw/main/ssd_mobilenet_v1_12.onnx'
}

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

MODEL_CONFIGS = {
    'nanodet': {
        'class': NanoDet,
        'input_size': (416, 416),
        'params': {
            'probThreshold': 0.35,
            'iouThreshold': 0.6
        }
    },
    'yolox': {
        'class': YoloX,
        'input_size': (640, 640),
        'params': {
            'confThreshold': 0.5,
            'nmsThreshold': 0.5,
            'objThreshold': 0.5
        }
    },
    'ssd': {
        'class': SSD,
        'input_size': (300, 300),
        'params': {
            'confThreshold': 0.5,
            'nmsThreshold': 0.4
        }
    }
}