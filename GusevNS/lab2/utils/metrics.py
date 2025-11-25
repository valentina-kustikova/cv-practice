import numpy as np


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def match_detections(detections, truths, iou_threshold):
    matches = []
    used_truths = set()
    for detection in detections:
        best_iou = 0.0
        best_idx = -1
        det_label = detection.label.lower()
        for idx, truth in enumerate(truths):
            if idx in used_truths:
                continue
            if truth.class_name != det_label:
                continue
            current_iou = iou(
                (
                    detection.box[0],
                    detection.box[1],
                    detection.box[0] + detection.box[2],
                    detection.box[1] + detection.box[3],
                ),
                truth.box,
            )
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matches.append((detection, truths[best_idx]))
            used_truths.add(best_idx)
    tp = len(matches)
    fp = len(detections) - tp
    fn = len(truths) - tp
    return tp, fp, fn


def compute_metrics(aggregated):
    tp = aggregated["tp"]
    fp = aggregated["fp"]
    fn = aggregated["fn"]
    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    fdr = fp / (fp + tp) if fp + tp > 0 else 0.0
    return tpr, fdr

