def calculate_iou(box_a, box_b):
    """
    Считает Intersection over Union между двумя прямоугольниками.
    box: (x_min, y_min, x_max, y_max)
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    
    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def match_frame_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Сопоставляет предсказания с разметкой для одного кадра.
    Логика строго соответствует заданию:
    1. Учитываем только машины и автобусы.
    2. Сортируем предсказания по уверенности.
    3. Жадный поиск соответствий (Greedy Matching).
    """
    target_classes = {'car', 'bus'}

    # Фильтрация классов
    gt_filtered = [g for g in gt_boxes if g['class'].lower() in target_classes]
    pred_filtered = [p for p in pred_boxes if p['class'].lower() in target_classes]

    if not gt_filtered and not pred_filtered:
        return 0, 0, 0

    # Сортировка предсказаний
    pred_filtered.sort(key=lambda x: x['conf'], reverse=True)

    tp = 0
    fp = 0
    
    gt_is_matched = [False] * len(gt_filtered)

    # Проход по предсказаниям
    for pred in pred_filtered:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_filtered):
            if gt_is_matched[i]:
                continue
            
            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        # Проверяем порог
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_is_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = len(gt_filtered) - sum(gt_is_matched)

    return tp, fp, fn

def calculate_final_metrics(all_gt_data, all_pred_data, iou_threshold=0.5):
    """
    Считает итоговые TPR и FDR по всему датасету.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gt, pred in zip(all_gt_data, all_pred_data):
        tp, fp, fn = match_frame_predictions(gt, pred, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Расчет TPR (Recall) = TP / (TP + FN)
    if (total_tp + total_fn) > 0:
        tpr = total_tp / (total_tp + total_fn)
    else:
        tpr = 0.0

    # Расчет FDR = FP / (TP + FP)
    if (total_tp + total_fp) > 0:
        fdr = total_fp / (total_tp + total_fp)
    else:
        fdr = 0.0

    return tpr, fdr, total_tp, total_fp, total_fn