import os
import urllib.request
import sys

MODELS = [
    # 1. YOLOv8 (ONNX)
    {
        'name': 'YOLOv8',
        'files': [
            {
                'url': 'https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8l.onnx',
                'filename': 'yolov8l.onnx'
            }
        ]
    },
    # 2. YOLOv4 (Darknet)
    {
        'name': 'YOLOv4',
        'files': [
            {
                'url': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
                'filename': 'yolov4.weights'
            },
            {
                'url': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
                'filename': 'yolov4.cfg'
            }
        ]
    },
    # 3. NanoDet-Plus (ONNX)
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
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[✓] Сохранено: {dest}")
    except Exception as e:
        print(f"[X] Ошибка скачивания {url}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit(1)

if __name__ == '__main__':
    print(f"Папка для моделей: {MODELS_DIR}")
    for model in MODELS:
        print(f"\nПроверка модели: {model['name']}")
        for file in model['files']:
            dest = os.path.join(MODELS_DIR, file['filename'])
            download_file(file['url'], dest)
    print("\nВсе модели проверены и готовы к работе!")