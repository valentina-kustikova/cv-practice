# src/metrics.py

import numpy as np

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def calculate_tpr_fdr(detections, ground_truths, iou_threshold=0.5):
    if len(ground_truths) == 0:
        if len(detections) == 0:
             return 1.0, 0.0 # Нет объектов - все "нашли"
        else:
             return 0.0, 1.0 # Нашли то, чего нет

    if len(detections) == 0:
        # Не нашли, что нужно, но и ложно не находили
        return 0.0, 0.0

    detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for det in detections_sorted:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(ground_truths):
            if gt['class'] == det['class'] and i not in matched_gt:
                iou = calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

    false_negatives = len(ground_truths) - true_positives

    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    fdr = false_positives / (false_positives + true_positives) if (false_positives + true_positives) > 0 else 0.0

    return tpr, fdr
