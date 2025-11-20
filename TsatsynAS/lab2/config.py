MODELS_CONFIG = {
    'yolo': {
        'model': 'models/yolov4.weights',
        'config': 'models/yolov4.cfg',
        'classes': 'models/coco.names',
        'detector_class': 'YOLODetector'
    },
    'ssd': {
        'model': 'models/mobilenet_iter_73000.caffemodel',
        'config': 'models/deploy.prototxt',
        'classes': 'models/coco.names',
        'detector_class': 'SSDDetector'
    },
    'faster_rcnn': {
        'model': 'models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb',
        'config': 'models/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt',
        'classes': 'models/coco.names',
        'detector_class': 'FasterRCNNDetector'
    }

}

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane']

# Colors for different vehicle classes
VEHICLE_COLORS = {
    'car': (0, 255, 0),        # Green
    'bus': (255, 0, 0),        # Blue
    'truck': (0, 0, 255),      # Red
    'motorcycle': (255, 255, 0),  # Cyan
    'bicycle': (255, 0, 255),  # Magenta
    'train': (0, 255, 255),    # Yellow
    'boat': (128, 0, 128),     # Purple
    'airplane': (255, 165, 0)  # Orange
}
