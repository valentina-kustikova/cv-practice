import cv2
import os

def load_images(folder):
    """Загружает все изображения из указанной папки."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Папка не найдена: {folder}")
    if not os.path.isdir(folder):
        raise ValueError(f"Путь не является папкой: {folder}")
    
    images = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is not None:
                images[fname] = img
    return images



def load_annotations(path):
    """Загружает аннотации из файла. Формат: frame_id class x1 y1 x2 y2"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл аннотаций не найден: {path}")
    
    ann = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            class_name = parts[1]
            x1, y1, x2, y2 = map(int, parts[2:6])
            fname = f"{frame_id:06d}.jpg"
            if fname not in ann:
                ann[fname] = []
            ann[fname].append({'class': class_name, 'bbox': [x1, y1, x2, y2]})
    return ann



def draw_boxes(image, detections):
    COLORS = {
        'CAR': (0, 255, 0),      # Зеленый
        'BUS': (255, 0, 0),      # Синий
        'MOTORCYCLE': (0, 0, 255),# Красный
        'TRUCK': (255, 255, 0)   # Голубой
    }
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class'].upper()
        conf = det['confidence']
        color = COLORS.get(class_name, (255, 255, 255))
        # Прямоугольник
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Подпись: класс и confidence
        label = f"{class_name} {conf:.3f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img


def iou(boxA, boxB):
    """Вычисляет Intersection over Union (IoU) для двух bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / float(union)

def normalize_class_name(class_name):
    """Нормализует название класса для сравнения: приводит к верхнему регистру."""
    return class_name.upper()

def calculate_metrics(detections, gts, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    for dets, gt_boxes in zip(detections, gts):
        matched = set()
        for det in dets:
            found = False
            det_class = normalize_class_name(det['class'])
            for i, gt in enumerate(gt_boxes):
                gt_class = normalize_class_name(gt['class'])
                if gt_class == det_class and i not in matched and iou(det['bbox'], gt['bbox']) > iou_threshold:
                    TP += 1
                    matched.add(i)
                    found = True
                    break
            if not found:
                FP += 1
        FN += len(gt_boxes) - len(matched)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    return TPR, FDR
