import os
import urllib.request
import tarfile
import shutil

# Структура папок
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ссылки на файлы
URLS = {
    "ssd_archive": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
    "ssd_config": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
    
    "rcnn_archive": "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz",
    "rcnn_config": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt",
    
    "yolo_weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "yolo_config": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    
    "coco_names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

def download_file(url, dest_path):
    print(f"Скачивание {url}...")
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"-> Сохранено в {dest_path}")
    except Exception as e:
        print(f"ОШИБКА при скачивании {url}: {e}")

def extract_tar(tar_path, extract_to):
    print(f"Распаковка {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print("-> Распаковка завершена")
    except Exception as e:
        print(f"ОШИБКА при распаковке: {e}")

def setup():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # --- 1. Настройка SSD MobileNet ---
    print("\n=== Настройка SSD MobileNet V2 ===")
    ssd_dir = os.path.join(MODELS_DIR, "ssd_mobilenet_v2_coco_2018_03_29")
    if not os.path.exists(ssd_dir):
        os.makedirs(ssd_dir)
    
    # Скачиваем архив
    tar_path = os.path.join(MODELS_DIR, "ssd.tar.gz")
    download_file(URLS["ssd_archive"], tar_path)
    extract_tar(tar_path, MODELS_DIR) # Распакуется в папку с именем как в архиве
    os.remove(tar_path) # Удаляем архив
    
    # Скачиваем конфиг .pbtxt
    download_file(URLS["ssd_config"], os.path.join(ssd_dir, "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"))

    # --- 2. Настройка Faster R-CNN ---
    print("\n=== Настройка Faster R-CNN ===")
    rcnn_dir = os.path.join(MODELS_DIR, "faster_rcnn_resnet50_coco_2018_01_28")
    if not os.path.exists(rcnn_dir):
        os.makedirs(rcnn_dir)
        
    tar_path = os.path.join(MODELS_DIR, "rcnn.tar.gz")
    download_file(URLS["rcnn_archive"], tar_path)
    extract_tar(tar_path, MODELS_DIR)
    os.remove(tar_path)
    
    download_file(URLS["rcnn_config"], os.path.join(rcnn_dir, "faster_rcnn_resnet50_coco_2018_01_28.pbtxt"))

    # --- 3. Настройка YOLO ---
    print("\n=== Настройка YOLOv4-tiny ===")
    yolo_dir = os.path.join(MODELS_DIR, "yolo")
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)
    
    download_file(URLS["yolo_weights"], os.path.join(yolo_dir, "yolov4-tiny.weights"))
    download_file(URLS["yolo_config"], os.path.join(yolo_dir, "yolov4-tiny.cfg"))

    # --- 4. Общие файлы ---
    print("\n=== Скачивание списка классов ===")
    download_file(URLS["coco_names"], os.path.join(MODELS_DIR, "coco_names.txt"))

    print("\n\nГОТОВО! Все модели загружены.")

if __name__ == "__main__":
    setup()