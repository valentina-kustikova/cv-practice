import os


# ============================================================================
# Модуль вспомогательных инструментов (Utilities):
# - compute_iou: функция расчета метрики Intersection over Union
# - Metrics: класс для накопления статистики (TP, FP, FN) и расчета TPR/FDR
# - parse_ground_truth: функция парсинга текстовых файлов разметки
# Отвечает за математику оценки качества и чтение данных с диска
# ============================================================================

def compute_iou(boxA, boxB):
    """
    Считает Intersection over Union (IoU) между двумя боксами.
    boxA, boxB: [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class Metrics:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, predictions, ground_truths, iou_threshold=0.5):
        """
        predictions: список [label_str, confidence, x, y, w, h]
        ground_truths: список [label_str, x, y, w, h]
        """
        matched_gt = set()

        for pred in predictions:
            p_label = pred[0]
            p_box = pred[2:]

            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(ground_truths):
                gt_label = gt[0]
                # Сравниваем классы (приводим к нижнему регистру: "CAR" == "car")
                if p_label.lower() == gt_label.lower():
                    iou = compute_iou(p_box, gt[1:])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                self.TP += 1
                matched_gt.add(best_gt_idx)
            else:
                self.FP += 1

        # Все GT, которые не нашли пару, считаются FN
        self.FN += len(ground_truths) - len(matched_gt)

    def get_metrics(self):
        tpr = self.TP / (self.TP + self.FN + 1e-6)
        fdr = self.FP / (self.TP + self.FP + 1e-6)
        return tpr, fdr


def parse_ground_truth(filepath):
    """
    Парсит файл формата: Frame_ID Class Xmin Ymin Xmax Ymax
    Возвращает словарь: { frame_number: [ ['CAR', x, y, w, h], ... ] }
    """
    gt_dict = {}
    if not os.path.exists(filepath):
        print(f"Внимание: файл разметки не найден: {filepath}")
        return gt_dict

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue

            try:
                frame_id = int(parts[0])
                cls_name = parts[1]
                xmin = int(parts[2])
                ymin = int(parts[3])
                xmax = int(parts[4])
                ymax = int(parts[5])

                w = xmax - xmin
                h = ymax - ymin

                if frame_id not in gt_dict:
                    gt_dict[frame_id] = []

                gt_dict[frame_id].append([cls_name, xmin, ymin, w, h])
            except ValueError:
                continue

    print(f"Загружена разметка для {len(gt_dict)} кадров.")
    return gt_dict
