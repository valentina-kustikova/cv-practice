def calculate_iou(boxA, boxB):
    """Вычисление Intersection over Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    if boxAArea + boxBArea - interArea == 0: return 0
    return interArea / float(boxAArea + boxBArea - interArea)


def compute_metrics(ground_truths, detections, iou_threshold=0.5):
    """Возвращает TPR, FDR и сырые счетчики (TP, FP, FN)."""
    tp = 0
    fp = 0
    matched_gt = [False] * len(ground_truths)

    for det in detections:
        box_det = det['box']
        best_iou = 0
        best_gt_idx = -1

        for i, box_gt in enumerate(ground_truths):
            iou = calculate_iou(box_det, box_gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
            tp += 1
            matched_gt[best_gt_idx] = True
        else:
            fp += 1

    fn = len(ground_truths) - tp
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return tpr, fdr, tp, fp, fn


def parse_ground_truth(file_path):
    """Чтение файла разметки mov03478.txt."""
    gt_data = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6: continue
                frame_id = int(parts[0])
                x_min = int(parts[2])
                y_min = int(parts[3])
                w = int(parts[4]) - x_min
                h = int(parts[5]) - y_min

                if frame_id not in gt_data:
                    gt_data[frame_id] = []
                gt_data[frame_id].append((x_min, y_min, w, h))
    except FileNotFoundError:
        print(f"Файл разметки {file_path} не найден.")
    return gt_data
