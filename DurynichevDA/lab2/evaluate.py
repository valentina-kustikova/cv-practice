import numpy as np

def iou(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter_area

    return inter_area / union

def evaluate_frame(gt_boxes, pred_boxes, iou_thresh=0.5):
    tp = fp = 0
    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            curr_iou = iou(pred['box'], gt['box'])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_idx = idx
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn