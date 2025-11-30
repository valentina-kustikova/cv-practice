from typing import List, Tuple
from detector_base import Detection


class DetectionEvaluator:
    """
    TPR и FDR по классу car
    """

    def __init__(self, target_class: str = "car"):
        self.target_class = target_class.lower()
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int],
             box_b: Tuple[int, int, int, int]) -> float:
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        x_left = max(x1_a, x1_b)
        y_top = max(y1_a, y1_b)
        x_right = min(x2_a, x2_b)
        y_bottom = min(y2_a, y2_b)

        inter_w = max(0, x_right - x_left)
        inter_h = max(0, y_bottom - y_top)
        intersection = inter_w * inter_h

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def evaluate_frame(self,
                       detections: List[Detection],
                       gt_boxes: List[Tuple[int, int, int, int]],
                       iou_threshold: float = 0.5) -> Tuple[float, float]:
        """
        Обновляем счётчики + возвращаем TPR/FDR для кадра.
        """
        frame_tp = 0
        frame_fp = 0
        frame_fn = 0

        gt_used = [False] * len(gt_boxes)

        candidate_detections = sorted(
            [d for d in detections if d.class_name.lower() == self.target_class],
            key=lambda d: d.confidence,
            reverse=True
        )

        for det in candidate_detections:
            best_iou = 0.0
            best_idx = -1

            for idx, gt in enumerate(gt_boxes):
                if gt_used[idx]:
                    continue
                iou = self._iou(det.bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx != -1:
                frame_tp += 1
                self.true_positives += 1
                gt_used[best_idx] = True
            else:
                frame_fp += 1
                self.false_positives += 1

        unused_gt = sum(1 for used in gt_used if not used)
        frame_fn += unused_gt
        self.false_negatives += unused_gt

        frame_tpr = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0.0
        frame_fdr = frame_fp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0.0

        return frame_tpr, frame_fdr

    def get_metrics(self) -> Tuple[float, float]:
        """TPR и FDR по всем кадрам"""
        tpr = self.true_positives / (self.true_positives + self.false_negatives) \
            if (self.true_positives + self.false_negatives) > 0 else 0.0
        fdr = self.false_positives / (self.true_positives + self.false_positives) \
            if (self.true_positives + self.false_positives) > 0 else 0.0
        return tpr, fdr

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
