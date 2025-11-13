def calculate_metrics(detections, ground_truth, iou_threshold=0.5):
    """
    Calculate TPR (True Positive Rate) and FDR (False Discovery Rate)
    """
    if len(ground_truth) == 0:
        if len(detections) == 0:
            return 1.0, 0.0  # Perfect if no detections and no ground truth
        else:
            return 0.0, 1.0  # All false positives if no ground truth but detections exist

    true_positives = 0
    false_positives = 0

    matched_gt = set()

    for det in detections:
        det_bbox = det['bbox']
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            gt_bbox = gt['bbox']
            iou = calculate_iou(det_bbox, gt_bbox)

            if iou > best_iou and det['class_name'] == gt['class_name']:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched_gt)

    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    fdr = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    return tpr, fdr


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
