import numpy as np

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def compute_tpr_fdr(det_boxes, det_labels, gt_boxes, gt_labels, confs=None, iou_thr=0.5):
    
    sorted_indices = np.argsort(confs)[::-1]
    det_boxes = [det_boxes[i] for i in sorted_indices]
    det_labels = [det_labels[i] for i in sorted_indices]
    confs = [confs[i] for i in sorted_indices]
    
    tp = fp = fn = 0
    matched = [False] * len(gt_boxes)
    for db, dl in zip(det_boxes, det_labels):
        found = False
        for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if not matched[j] and dl == gl and box_iou(db, gb) >= iou_thr:
                tp += 1
                matched[j] = True
                found = True
                break
        if not found:
            fp += 1
    fn = sum(1 for m in matched if not m)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return tpr, fdr