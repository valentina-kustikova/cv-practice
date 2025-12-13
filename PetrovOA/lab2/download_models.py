"""
Скрипт для скачивания моделей для детекции объектов.

Скачивает:
1. YOLOv8l (ONNX) - Ultralytics
2. SSD MobileNet V1 (TensorFlow) - TF Object Detection API
3. NanoDet-Plus (ONNX) - OpenCV Zoo
"""

import os
import sys
import urllib.request
import tarfile
import hashlib
from pathlib import Path


# Директория для моделей
MODELS_DIR = "models"

# Информация о моделях
MODELS = {
    "yolov8l_onnx": {
        "url": "https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8l.onnx",
        "filename": "yolov8l.onnx",
        "description": "YOLOv8l (ONNX, готовый к использованию)"
    },
    "ssd_mobilenet_v1": {
        "url": "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz",
        "filename": "ssd_mobilenet_v1.tar.gz",
        "description": "SSD MobileNet V1 COCO (TensorFlow)",
        "extract": True,
        "model_file": "ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb"
    },
    "ssd_mobilenet_v1_pbtxt": {
        "url": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v1_coco_2017_11_17.pbtxt",
        "filename": "ssd_mobilenet_v1_coco.pbtxt",
        "description": "SSD MobileNet V1 config (pbtxt для OpenCV DNN)"
    },
    "nanodet_plus": {
        "url": "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx",
        "filename": "nanodet-plus-m_416.onnx",
        "description": "NanoDet-Plus-m 416x416 (ONNX)"
    }
}

# pbtxt конфигурация для SSD
SSD_PBTXT_CONTENT = '''
item {
  id: 1
  name: 'person'
}
item {
  id: 2
  name: 'bicycle'
}
item {
  id: 3
  name: 'car'
}
item {
  id: 4
  name: 'motorcycle'
}
item {
  id: 5
  name: 'airplane'
}
item {
  id: 6
  name: 'bus'
}
item {
  id: 7
  name: 'train'
}
item {
  id: 8
  name: 'truck'
}
item {
  id: 9
  name: 'boat'
}
item {
  id: 10
  name: 'traffic light'
}
'''


def download_file(url: str, dest_path: str, description: str = "") -> bool:
    """
    Скачивание файла с отображением прогресса.
    
    Args:
        url: URL для скачивания
        dest_path: путь для сохранения
        description: описание файла
        
    Returns:
        success: True если скачивание успешно
    """
    print(f"\n{'='*60}")
    print(f"Скачивание: {description}")
    print(f"URL: {url}")
    print(f"Сохранение: {dest_path}")
    print('='*60)
    
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rПрогресс: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()
            else:
                mb_downloaded = downloaded / (1024 * 1024)
                sys.stdout.write(f"\rСкачано: {mb_downloaded:.1f} MB")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print("\n✓ Скачивание завершено!")
        return True
        
    except Exception as e:
        print(f"\n✗ Ошибка скачивания: {e}")
        return False


def extract_tar_gz(archive_path: str, extract_dir: str) -> bool:
    """
    Распаковка tar.gz архива.
    
    Args:
        archive_path: путь к архиву
        extract_dir: директория для распаковки
        
    Returns:
        success: True если распаковка успешна
    """
    print(f"Распаковка {archive_path}...")
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("✓ Распаковка завершена!")
        return True
    except Exception as e:
        print(f"✗ Ошибка распаковки: {e}")
        return False


def create_ssd_pbtxt(models_dir: str) -> str:
    """
    Создание pbtxt файла для SSD модели.
    Используем готовый файл из opencv_extra или создаём минимальный.
    """
    pbtxt_path = os.path.join(models_dir, "ssd_inception_v2_coco.pbtxt")
    
    # Минимальная конфигурация для OpenCV DNN
    pbtxt_content = '''node {
  name: "image_tensor"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_UINT8
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: -1
        }
        dim {
          size: -1
        }
        dim {
          size: 3
        }
      }
    }
  }
}
'''
    
    # Скачиваем готовый pbtxt из opencv_extra
    pbtxt_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_inception_v2_coco_2017_11_17.pbtxt"
    
    print(f"\nСкачивание конфигурации SSD...")
    try:
        urllib.request.urlretrieve(pbtxt_url, pbtxt_path)
        print(f"✓ Конфигурация сохранена: {pbtxt_path}")
    except Exception as e:
        print(f"Предупреждение: не удалось скачать pbtxt: {e}")
        print("Будет использован локальный файл")
        with open(pbtxt_path, 'w') as f:
            f.write(pbtxt_content)
    
    return pbtxt_path


def main():
    """Основная функция скачивания моделей."""
    print("="*60)
    print("Скрипт скачивания моделей для детекции объектов")
    print("="*60)
    
    # Создаём директорию для моделей
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\nДиректория для моделей: {os.path.abspath(MODELS_DIR)}")
    
    # Выбор моделей для скачивания
    print("\nДоступные модели:")
    for i, (key, info) in enumerate(MODELS.items(), 1):
        print(f"  {i}. {key}: {info['description']}")
    
    print("\nКакие модели скачать?")
    print("  1 - Только YOLOv8l (ONNX)")
    print("  2 - Только SSD MobileNet V1 (TensorFlow)")
    print("  3 - Только NanoDet-Plus")
    print("  4 - Все модели")
    print("  0 - Выход")
    
    try:
        choice = input("\nВаш выбор (1-4, или 0 для выхода): ").strip()
    except KeyboardInterrupt:
        print("\nОтменено пользователем")
        return
    
    models_to_download = []
    
    if choice == '1':
        models_to_download = ['yolov8l_onnx']
    elif choice == '2':
        models_to_download = ['ssd_mobilenet_v1', 'ssd_mobilenet_v1_pbtxt']
    elif choice == '3':
        models_to_download = ['nanodet_plus']
    elif choice == '4':
        models_to_download = ['yolov8l_onnx', 'ssd_mobilenet_v1', 'ssd_mobilenet_v1_pbtxt', 'nanodet_plus']
    elif choice == '0':
        print("Выход")
        return
    else:
        print("Неверный выбор")
        return
    
    # Скачивание выбранных моделей
    for model_key in models_to_download:
        model_info = MODELS[model_key]
        dest_path = os.path.join(MODELS_DIR, model_info['filename'])
        
        # Проверяем, существует ли файл
        if os.path.exists(dest_path):
            print(f"\n⚠ Файл уже существует: {dest_path}")
            response = input("Перезаписать? (y/n): ").strip().lower()
            if response != 'y':
                print("Пропускаем...")
                continue
        
        # Скачиваем
        success = download_file(model_info['url'], dest_path, model_info['description'])
        
        if not success:
            continue
        
        # Распаковка если нужно
        if model_info.get('extract'):
            extract_tar_gz(dest_path, MODELS_DIR)
            print(f"✓ Модель распакована в: {MODELS_DIR}/ssd_mobilenet_v1_coco_2017_11_17/")
        
        # Выводим примечание если есть
        if 'note' in model_info:
            print(f"\n⚠ Примечание: {model_info['note']}")
    
    print("\n" + "="*60)
    print("Скачивание завершено!")
    print("="*60)
    
    # Выводим информацию об использовании
    print("\nПути к моделям для использования:")
    
    if 'yolov8l_onnx' in models_to_download:
        print(f"\n  YOLOv8l:")
        print(f"    --model yolov8 --model_path {MODELS_DIR}/yolov8l.onnx")
    
    if 'ssd_mobilenet_v1' in models_to_download:
        print(f"\n  SSD MobileNet V1:")
        print(f"    --model ssd --model_path {MODELS_DIR}/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb --config_path {MODELS_DIR}/ssd_mobilenet_v1_coco.pbtxt")
    
    if 'nanodet_plus' in models_to_download:
        print(f"\n  NanoDet-Plus:")
        print(f"    --model nanodet --model_path {MODELS_DIR}/nanodet-plus-m_416.onnx")
    
    print("\nПример запуска:")
    print(f"  python main.py --data_path ./data --labels_path labels.txt --model yolov8 --model_path {MODELS_DIR}/yolov8l.onnx --show")


if __name__ == "__main__":
    main()
