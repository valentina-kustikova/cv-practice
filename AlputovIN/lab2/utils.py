import cv2
import os
import re
import numpy as np

def load_images(folder):
    """Загружает все изображения из указанной папки."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Папка не найдена: {folder}")
    if not os.path.isdir(folder):
        raise ValueError(f"Путь не является папкой: {folder}")
    
    images = {}
    files = sorted(os.listdir(folder))
    for fname in files:
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is not None:
                images[fname] = img
    return images


def get_frame_id_from_filename(filename):
    """
    Пытается извлечь числовой ID из имени файла.
    """
    numbers = re.findall(r'\d+', filename)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            return -1
    return -1


def load_annotations(path, image_folder=None):
    """
    Загружает аннотации из файла. 
    Предполагаемый формат в файле: frame_id class x y w h
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл аннотаций не найден: {path}")
    
    # 1. Строим карту имен файлов
    id_to_filename = {}
    if image_folder and os.path.exists(image_folder):
        for fname in os.listdir(image_folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                fid = get_frame_id_from_filename(fname)
                if fid != -1:
                    id_to_filename[fid] = fname
    
    ann = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            try:
                frame_id = int(parts[0])
                class_name = parts[1]
                # Читаем как x, y, w, h (стандарт для большинства датасетов)
                val1, val2, val3, val4 = map(int, parts[2:6])
                
                # ЛОГИКА ОПРЕДЕЛЕНИЯ КООРДИНАТ
                # Если 3-е число меньше 1-го (например x2 < x1), то это явно ширина/высота
                # Либо просто всегда считаем, что это x, y, w, h (безопаснее для этой лабы)
                x1 = val1
                y1 = val2
                x2 = val1 + val3 # x + w
                y2 = val2 + val4 # y + h
                
            except ValueError:
                continue

            target_filenames = []
            if frame_id in id_to_filename:
                target_filenames.append(id_to_filename[frame_id])
            else:
                target_filenames = [
                    f"{frame_id:06d}.jpg",
                    f"{frame_id}.jpg",
                    f"frame_{frame_id}.jpg",
                    f"img{frame_id:05d}.jpg"
                ]

            bbox_data = {'class': class_name, 'bbox': [x1, y1, x2, y2]}

            for fname in target_filenames:
                if fname not in ann:
                    ann[fname] = []
                if bbox_data not in ann[fname]:
                    ann[fname].append(bbox_data)
                    
    return ann


def draw_boxes(image, detections):
    COLORS = {
        'CAR': (0, 255, 0), 'BUS': (255, 0, 0),
        'MOTORCYCLE': (0, 0, 255), 'TRUCK': (255, 255, 0)
    }
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class'].upper()
        conf = det['confidence']
        color = COLORS.get(class_name, (255, 255, 255))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.3f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img


def iou(boxA, boxB):
    # Важно: boxA и boxB должны быть [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union = boxAArea + boxBArea - interArea
    if union <= 0: return 0.0
    return interArea / float(union)

def normalize_class_name(class_name):
    return class_name.upper()

def calculate_metrics(detections, gts, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    for dets, gt_boxes in zip(detections, gts):
        matched_gt = set()
        for det in dets:
            found = False
            det_class = normalize_class_name(det['class'])
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt: continue
                gt_class = normalize_class_name(gt['class'])
                
                if gt_class == det_class and iou(det['bbox'], gt['bbox']) > iou_threshold:
                    TP += 1
                    matched_gt.add(i)
                    found = True
                    break
            if not found:
                FP += 1
        FN += len(gt_boxes) - len(matched_gt)
    
    denom_tpr = TP + FN
    TPR = TP / denom_tpr if denom_tpr > 0 else 0.0
    denom_fdr = TP + FP
    FDR = FP / denom_fdr if denom_fdr > 0 else 0.0
    return TPR, FDR