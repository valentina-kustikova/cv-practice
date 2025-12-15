import os
import urllib.request

MODELS = [
    # 1. YOLOv8 (Petrov)
    {
        'name': 'YOLOv8',
        'files': [
            {
                'url': 'https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8l.onnx',
                'filename': 'yolov8l.onnx'
            }
        ]
    },
    # 2. Faster R-CNN (Korotin)
    {
        'name': 'Faster R-CNN',
        'files': [
            {
                'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
                'filename': 'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'
            },
            {
                'url': 'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt',
                'filename': 'faster_rcnn_inception_v2_coco.pbtxt'
            }
        ]
    },
    # 3. NanoDet-Plus (Smirnov)
    {
        'name': 'NanoDet-Plus',
        'files': [
            {
                'url': 'https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx',
                'filename': 'nanodet-plus-m_416.onnx'
            }
        ]
    }
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'detectors', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"[✓] {os.path.basename(dest)} уже загружен.")
        return
    print(f"[→] Скачивание {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[✓] Сохранено: {dest}")

if __name__ == '__main__':
    for model in MODELS:
        print(f"\nМодель: {model['name']}")
        for file in model['files']:
            dest = os.path.join(MODELS_DIR, file['filename'])
            download_file(file['url'], dest)
    print("\nВсе модели скачаны!")
