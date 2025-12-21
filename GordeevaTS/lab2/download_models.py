import os
import urllib.request

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Скачиваю {os.path.basename(dest_path)}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"  Готово: {os.path.basename(dest_path)}")
        except Exception as e:
            print(f"  Ошибка: {e}")
    else:
        print(f"Уже есть: {os.path.basename(dest_path)}")

def main():
    print("=" * 60)
    print("Скачивание моделей для детекции...")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    
    print("\n1. MobileNet-SSD (Caffe модель):")
    print("-" * 40)
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    download_file(prototxt_url, "models/deploy.prototxt")
    
    caffemodel_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdQdFk"
    caffemodel_path = "models/mobilenet_iter_73000.caffemodel"
    
    if not os.path.exists(caffemodel_path):
        print("Скачиваю mobilenet_iter_73000.caffemodel...")
        print("Это может занять некоторое время (~20MB)...")        
        alt_urls = [
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
            "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdQdFk&confirm=t",
        ]
        
        for url in alt_urls:
            try:
                urllib.request.urlretrieve(url, caffemodel_path)
                print(f"  Успешно скачано из: {url}")
                break
            except:
                continue
    
    print("\n2. YOLO модели:")
    print("-" * 40)
    
    yolov4_weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    download_file(yolov4_weights_url, "models/yolov4.weights")
    yolov4_cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
    download_file(yolov4_cfg_url, "models/yolov4.cfg")
    
    yolov4_tiny_weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    download_file(yolov4_tiny_weights_url, "models/yolov4-tiny.weights")
    yolov4_tiny_cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    download_file(yolov4_tiny_cfg_url, "models/yolov4-tiny.cfg")
    
    print("\n3. Файл классов COCO:")
    print("-" * 40)
    coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    download_file(coco_names_url, "models/coco.names")
    
    print("\n" + "=" * 60)
    print("Проверка скачанных файлов:")
    print("=" * 60)
    expected_files = [
        "models/deploy.prototxt",
        "models/mobilenet_iter_73000.caffemodel",
        "models/yolov4.weights",
        "models/yolov4.cfg",
        "models/yolov4-tiny.weights",
        "models/yolov4-tiny.cfg",
        "models/coco.names"
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # в MB
            print(f"✓ {os.path.basename(file)}: {size:.1f} MB")
        else:
            print(f"✗ {os.path.basename(file)}: НЕ НАЙДЕН")
    
    print("\n" + "=" * 60)
    print("Готово! Модели загружены.")
    print("  python main.py --images_dir data/images --annotations_dir data/annotations --model yolo --display")
    print("  python main.py --images_dir data/images --annotations_dir data/annotations --model ssd_mobilenet --display")
    print("  python main.py --images_dir data/images --annotations_dir data/annotations --model yolo_tiny --display")
    print("=" * 60)

if __name__ == "__main__":
    main()
