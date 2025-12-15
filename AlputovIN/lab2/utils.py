import cv2
import os
import re
import numpy as np

# Словарь для маппинга числовых ID в названия классов (для специфичных датасетов)
CLASS_ID_MAP = {
    '1': 'car', '2': 'bus', '3': 'truck', '4': 'motorcycle',
    '0': 'car', # Иногда 0
    'car': 'car', 'bus': 'bus', 'truck': 'truck', 'motorcycle': 'motorcycle'
}

def load_images(folder):
    """Загружает все изображения из указанной папки."""
    if not os.path.exists(folder):
        return {}

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
    numbers = re.findall(r'\d+', filename)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            return -1
    return -1


def load_annotations(path, image_folder=None):
    """
    Загружает аннотации. Пытается автоматически определить формат координат (xywh или xyxy).
    """
    if not os.path.exists(path):
        return {}

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
            # Обычно: frame_id, class, c1, c2, c3, c4
            if len(parts) < 6:
                continue

            try:
                frame_id = int(parts[0])
                raw_class = parts[1]

                # Маппинг класса (число -> строка)
                class_name = CLASS_ID_MAP.get(raw_class, raw_class)

                val1, val2, val3, val4 = map(float, parts[2:6]) # float для универсальности
                val1, val2, val3, val4 = int(val1), int(val2), int(val3), int(val4)

                # ЭВРИСТИКА ОПРЕДЕЛЕНИЯ ФОРМАТА
                # Если 3-е значение (x2/w) больше, чем 1-е значение (x1) + небольшой порог,
                # и при этом оно похоже на координату (большое число), то это скорее всего x2, а не ширина.
                # В видео датасетах ширина машины редко превышает координату X.

                if val3 > val1 and val4 > val2:
                    # Похоже на формат: x1, y1, x2, y2
                    x1, y1, x2, y2 = val1, val2, val3, val4
                else:
                    # Похоже на формат: x, y, w, h
                    x1, y1 = val1, val2
                    x2 = val1 + val3
                    y2 = val2 + val4

            except ValueError:
                continue

            target_filenames = []
            if frame_id in id_to_filename:
                target_filenames.append(id_to_filename[frame_id])
            else:
                target_filenames = [f"{frame_id:06d}.jpg", f"{frame_id}.jpg", f"img{frame_id:05d}.jpg"]

            # Валидация бокса
            if x2 <= x1 or y2 <= y1:
                continue

            bbox_data = {'class': class_name, 'bbox': [x1, y1, x2, y2]}

            for fname in target_filenames:
                if fname not in ann:
                    ann[fname] = []
                # Избегаем дублей
                if bbox_data not in ann[fname]:
                    ann[fname].append(bbox_data)

    return ann


def draw_boxes(image, detections, gt_boxes=None):
    """Рисует предсказания (зеленым/цветным) и GT (желтым пунктиром) для отладки."""
    COLORS = {
        'CAR': (0, 255, 0), 'BUS': (255, 0, 0),
        'MOTORCYCLE': (0, 0, 255), 'TRUCK': (255, 255, 0)
    }
    img = image.copy()

    # 1. Рисуем GT (Ground Truth) - Желтым
    if gt_boxes:
        for gt in gt_boxes:
            x1, y1, x2, y2 = gt['bbox']
            # Рисуем просто рамку
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 2. Рисуем Предсказания
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class'].upper()
        conf = det['confidence']
        color = COLORS.get(class_name, (0, 255, 0)) # Default green

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def iou(boxA, boxB):
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
    # Приводим к верхнему регистру и убираем лишнее
    return str(class_name).upper().strip()

def calculate_metrics(detections, gts, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0

    # Списки для дебага (можно раскомментировать print внутри)

    for dets, gt_boxes in zip(detections, gts):
        matched_gt = set()

        for det in dets:
            found = False
            det_class = normalize_class_name(det['class'])

            for i, gt in enumerate(gt_boxes):
                if i in matched_gt: continue
                gt_class = normalize_class_name(gt['class'])

                # Сверяем классы и IoU
                # Можно ослабить проверку классов, если названия совсем разные
                if gt_class == det_class:
                    score = iou(det['bbox'], gt['bbox'])
                    if score > iou_threshold:
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
