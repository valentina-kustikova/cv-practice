MODELS = {
    'yolo': {
        'model': 'models/yolov4.weights',
        'config': 'models/yolov4.cfg',
        'classes': 'models/coco.names'
    },
    'ssd_mobilenet': {
        'model': 'models/mobilenet_iter_73000.caffemodel',
        'config': 'models/deploy.prototxt',
        'classes': 'models/coco.names'
    },
    'yolo_tiny': {
        'model': 'models/yolov4-tiny.weights',
        'config': 'models/yolov4-tiny.cfg',
        'classes': 'models/coco.names'
    }
}

VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
