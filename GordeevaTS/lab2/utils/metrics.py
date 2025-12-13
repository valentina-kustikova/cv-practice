import numpy as np

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(detections, ground_truths, iou_threshold=0.5):
    if len(ground_truths) == 0:
        tp = 0
        fp = len(detections)
        fn = 0
    elif len(detections) == 0:
        tp = 0
        fp = 0
        fn = len(ground_truths)
    else:
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        tp = 0
        fp = 0
        used_gt = [False] * len(ground_truths)

        for det in detections:
            det_box = det['bbox']
            det_class = det['class_name']
            best_iou = 0
            best_idx = -1

            for i, gt in enumerate(ground_truths):
                if used_gt[i]:
                    continue
                if gt['class_name'] != det_class:
                    continue
                iou = calculate_iou(det_box, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold and best_idx != -1:
                used_gt[best_idx] = True
                tp += 1
            else:
                fp += 1

        fn = len([x for x in used_gt if not x]) 

    total_positives = tp + fn
    total_detections = tp + fp

    tpr = tp / total_positives if total_positives > 0 else 0.0
    fdr = fp / total_detections if total_detections > 0 else 0.0

    return tpr, fdr