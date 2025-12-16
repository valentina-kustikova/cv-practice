import cv2
import numpy as np
import os
import urllib.request
import sys
from annotation_parser import AnnotationParser


def load_ground_truth_new_format(annotation_parser, frame_number, image_width, image_height):
    """Загрузка ground truth для нового формата разметки"""
    return annotation_parser.get_ground_truth_for_frame(frame_number, image_width, image_height)


def load_ground_truth(annotation_path, image_width, image_height, class_names=None):
    """Загрузка ground truth разметки (старый формат - для обратной совместимости)"""
    ground_truth = []

    if not os.path.exists(annotation_path):
        return ground_truth

    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return ground_truth

            lines = content.split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * image_width
                        y_center = float(parts[2]) * image_height
                        width = float(parts[3]) * image_width
                        height = float(parts[4]) * image_height

                        # Проверка валидности координат
                        if (x_center < 0 or x_center > image_width or
                                y_center < 0 or y_center > image_height or
                                width <= 0 or height <= 0):
                            continue

                        x1 = max(0, int(x_center - width / 2))
                        y1 = max(0, int(y_center - height / 2))
                        x2 = min(image_width, int(x_center + width / 2))
                        y2 = min(image_height, int(y_center + height / 2))

                        # Проверка что bounding box валиден
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Получаем имя класса
                        class_name = None
                        if class_names:
                            class_name = class_names.get(class_id, f"class_{class_id}")
                        else:
                            class_name = map_class_id_to_vehicle(class_id)

                        ground_truth.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': 1.0
                        })
                    except (ValueError, IndexError) as e:
                        continue
    except Exception as e:
        print(f"Error reading annotation file {annotation_path}: {e}")

    return ground_truth


def map_class_id_to_vehicle(class_id):
    """Маппинг ID классов к названиям транспортных средств"""
    vehicle_mappings = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        1: 'bicycle',
        0: 'car',
        1: 'bus',
        2: 'truck',
        3: 'motorcycle',
        4: 'bicycle'
    }
    return vehicle_mappings.get(class_id, f"class_{class_id}")


def normalize_class_name(class_name):
    """Нормализация названий классов для сравнения"""
    class_name = class_name.upper().strip()

    # Маппинг различных вариантов написания
    class_mapping = {
        'CAR': 'car',
        'CARS': 'car',
        'AUTO': 'car',
        'AUTOMOBILE': 'car',
        'BUS': 'bus',
        'BUSES': 'bus',
        'TRUCK': 'truck',
        'TRUCKS': 'truck',
        'LORRY': 'truck',
        'MOTORCYCLE': 'motorcycle',
        'MOTORBIKE': 'motorcycle',
        'BIKE': 'motorcycle',
        'BICYCLE': 'bicycle',
        'CYCLE': 'bicycle'
    }

    return class_mapping.get(class_name, class_name.lower())


def calculate_detection_metrics(detections, ground_truth, iou_threshold=0.5, class_agnostic=True):
    """
    Вычисление метрик детектирования
    """
    # Если нет ground truth, возвращаем нулевые метрики
    if not ground_truth:
        return {
            'true_positives': 0,
            'false_positives': len(detections),
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_detections': len(detections),
            'total_ground_truth': 0,
            'matched_pairs': 0
        }

    # Если нет детекций
    if not detections:
        return {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(ground_truth),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_detections': 0,
            'total_ground_truth': len(ground_truth),
            'matched_pairs': 0
        }

    true_positives = 0
    false_positives = 0

    matched_gt = set()
    matched_detections = set()

    # Сопоставление детекций с ground truth
    for det_idx, detection in enumerate(detections):
        matched = False
        det_bbox = detection['bbox']
        det_class = normalize_class_name(detection['class_name']) if not class_agnostic else None

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            gt_bbox = gt['bbox']
            gt_class = normalize_class_name(gt['class_name']) if not class_agnostic else None

            # Проверка совпадения классов (если не class_agnostic)
            if not class_agnostic and det_class != gt_class:
                continue

            iou = calculate_iou(det_bbox, gt_bbox)

            if iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(gt_idx)
                matched_detections.add(det_idx)
                matched = True
                break

        if not matched:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched_gt)

    # Вычисление метрик
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_detections': len(detections),
        'total_ground_truth': len(ground_truth),
        'matched_pairs': len(matched_gt)
    }

    return metrics


def calculate_iou(bbox1, bbox2):
    """Вычисление Intersection over Union"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Вычисление координат пересечения
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)
    return iou




def draw_detections_with_metrics(image, detections, ground_truth=None, metrics=None, colors=None):
    """Отрисовка детекций и ground truth с метриками"""
    if colors is None:
        colors = {
            'detection': (0, 255, 0),  # Зеленый - детекции
            'ground_truth': (255, 0, 0),  # Синий - ground truth
            'true_positive': (0, 255, 0),  # Зеленый - true positive
            'false_positive': (0, 165, 255),  # Оранжевый - false positive
            'false_negative': (0, 0, 255),  # Красный - false negative
            'default': (255, 0, 255)  # Пурпурный
        }

    result_image = image.copy()

    # Сначала отрисовываем ground truth
    if ground_truth:
        for gt in ground_truth:
            x1, y1, x2, y2 = gt['bbox']
            class_name = gt.get('class_name', 'unknown')

            # Отрисовка ground truth
            cv2.rectangle(result_image, (x1, y1), (x2, y2), colors['ground_truth'], 2)
            label = f"GT: {class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), colors['ground_truth'], -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Затем отрисовываем детекции с цветами по типу
    for detection in detections:
        class_name = detection['class_name'].lower()
        confidence = detection['confidence']
        x1, y1, x2, y2 = detection['bbox']

        # Определяем цвет в зависимости от типа детекции
        # (это требует предварительного расчета метрик)
        color = colors['detection']

        # Отрисовка bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # Подпись с классом и уверенностью
        label = f"{class_name}: {confidence:.3f}"

        # Фон для текста
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)

        # Текст
        cv2.putText(result_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Добавление метрик на изображение
    if metrics:
        y_offset = 30
        metric_texts = [
            f"Precision: {metrics['precision']:.3f}",
            f"Recall: {metrics['recall']:.3f}",
            f"F1-Score: {metrics['f1_score']:.3f}",
            f"TP: {metrics['true_positives']} FP: {metrics['false_positives']} FN: {metrics['false_negatives']}",
            f"Detections: {metrics['total_detections']} GT: {metrics['total_ground_truth']}"
        ]

        for text in metric_texts:
            cv2.putText(result_image, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

    # Легенда
    legend_y = result_image.shape[0] - 90
    cv2.putText(result_image, "Blue: Ground Truth",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['ground_truth'], 1)
    cv2.putText(result_image, "Green: Detections",
                (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['detection'], 1)

    return result_image


def download_models():
    """Функция для загрузки предобученных моделей"""
    models_dir = "models_data"
    os.makedirs(models_dir, exist_ok=True)

    # YOLOv4-tiny
    yolo_files = {
        'weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
        'cfg': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
        'names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }

    print("Downloading YOLOv4-tiny model...")
    for name, url in yolo_files.items():
        ext = name
        filename = os.path.join(models_dir, f'yolov4-tiny.{ext}')
        if not os.path.exists(filename):
            print(f"  Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"  ✓ Downloaded {os.path.basename(filename)}")
            except Exception as e:
                print(f"  ✗ Error downloading {filename}: {e}")
                # Если не удалось скачать, создаем базовый файл классов
                if name == 'names':
                    create_default_coco_names(filename)
        else:
            print(f"  ✓ {os.path.basename(filename)} already exists")

    # Проверяем, что все файлы существуют
    check_model_files(models_dir)


def create_default_coco_names(filename):
    """Создание файла с классами COCO по умолчанию"""
    coco_classes = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    with open(filename, 'w') as f:
        for class_name in coco_classes:
            f.write(f"{class_name}\n")

    print(f"  ✓ Created default COCO names file: {filename}")


def check_model_files(models_dir):
    """Проверка существования всех необходимых файлов моделей"""
    required_files = [
        'yolov4-tiny.weights',
        'yolov4-tiny.cfg',
        'yolov4-tiny.names'
    ]

    print("\nChecking model files...")
    all_exists = True
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
            all_exists = False

    if all_exists:
        print("All model files are ready!")
    else:
        print("Some model files are missing. Please check the download.")

    return all_exists


def download_dataset(images_url=None, annotations_url=None, download_path="./dataset"):
    """Загрузка датасета с изображениями и разметкой"""
    import zipfile

    os.makedirs(download_path, exist_ok=True)

    # Если URL не предоставлены, используем стандартные
    if images_url is None:
        images_url = "https://cloud.unn.ru/s/nLkk7BXBqapNgcE/download"
    if annotations_url is None:
        annotations_url = "https://cloud.unn.ru/s/j4wA4nx8mZ4yfqD/download"

    # Загрузка изображений
    images_zip = os.path.join(download_path, "images.zip")
    images_extracted = os.path.join(download_path, "images")

    if not os.path.exists(images_extracted) or len(os.listdir(images_extracted)) == 0:
        print("Downloading images...")
        try:
            urllib.request.urlretrieve(images_url, images_zip)

            # Распаковка
            with zipfile.ZipFile(images_zip, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print("Images downloaded and extracted")

            # Удаление zip файла после распаковки
            if os.path.exists(images_zip):
                os.remove(images_zip)
        except Exception as e:
            print(f"Error downloading images: {e}")
    else:
        print("Images already exist")

    # Загрузка разметки
    annotations_zip = os.path.join(download_path, "annotations.zip")
    annotations_extracted = os.path.join(download_path, "annotations")

    if not os.path.exists(annotations_extracted) or len(os.listdir(annotations_extracted)) == 0:
        print("Downloading annotations...")
        try:
            urllib.request.urlretrieve(annotations_url, annotations_zip)

            # Распаковка
            with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print("Annotations downloaded and extracted")

            # Удаление zip файла после распаковки
            if os.path.exists(annotations_zip):
                os.remove(annotations_zip)
        except Exception as e:
            print(f"Error downloading annotations: {e}")
    else:
        print("Annotations already exist")

