def iou_xyxy(box1,
             box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def match_predictions_to_gt(
    gt_boxes,
    pred_boxes,
    iou_threshold = 0.5,
):
    vehicle_classes = {"car", "bus"}
    gt_filtered = [
        b for b in gt_boxes
        if b.class_name.lower() in vehicle_classes
    ]
    
    preds_filtered = [
        p for p in pred_boxes
        if p.class_name.lower() in vehicle_classes
    ]

    num_gt = len(gt_filtered)
    num_pred = len(preds_filtered)

    if num_gt == 0 and num_pred == 0:
        return 0, 0, 0

    gt_used = [False] * num_gt

    preds_sorted = sorted(
        enumerate(preds_filtered),
        key=lambda x: x[1].confidence,
        reverse=True,
    )

    tp = 0
    fp = 0

    for _, pred in preds_sorted:
        pred_box = pred.as_xyxy()

        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_filtered):
            if gt_used[i]:
                continue

            gt_box = gt.as_xyxy()
            iou = iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_used[best_gt_idx] = True
        else:
            fp += 1

    fn = num_gt - sum(gt_used)

    return tp, fp, fn


def compute_dataset_metrics(
    all_gt,
    all_pred,
    iou_threshold = 0.5,
):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        tp, fp, fn = match_predictions_to_gt(
            gt_boxes, pred_boxes, iou_threshold=iou_threshold
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom_pos = total_tp + total_fn
    if denom_pos == 0:
        tpr = 0.0
    else:
        tpr = total_tp / denom_pos

    denom_pred_pos = total_tp + total_fp
    if denom_pred_pos == 0:
        fdr = 0.0
    else:
        fdr = total_fp / denom_pred_pos

    return tpr, fdr, total_tp, total_fp, total_fn