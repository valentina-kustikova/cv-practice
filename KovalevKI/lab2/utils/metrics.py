import numpy as np

class FrameLevelDetectionEvaluator:
    def __init__(self, target_class="car", iou_threshold=0.5):
        self.target_class = target_class.lower()
        self.iou_thr = iou_threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def _xywh_to_xyxy(self, box):
        x, y, w, h = box
        return (x, y, x + w, y + h)

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def accumulate_frame(self, det_boxes, det_labels, det_confs, gt_boxes, gt_labels):
        sorted_dets = sorted(zip(det_boxes, det_labels, det_confs), key=lambda x: x[2], reverse=True)

        gt_boxes_xyxy = [self._xywh_to_xyxy(b) for b in gt_boxes]
        gt_used = [False] * len(gt_boxes)

        for box, label, conf in sorted_dets:
            if label.lower() != self.target_class:
                continue
            det_xyxy = self._xywh_to_xyxy(box)

            best_iou = 0.0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes_xyxy):
                if not gt_used[i]:
                    iou = self._calculate_iou(det_xyxy, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            if best_iou >= self.iou_thr and best_gt_idx != -1:
                self.tp += 1
                gt_used[best_gt_idx] = True
            else:
                self.fp += 1

        self.fn += sum(1 for used in gt_used if not used)

    def get_metrics(self):
        tpr = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        fdr = self.fp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        return tpr, fdr

    def reset(self):
        self.tp = self.fp = self.fn = 0