"""
Вспомогательные утилиты для детектирования объектов.
"""

from .nms import non_max_suppression, compute_iou
from .metrics import calculate_metrics, load_ground_truth
from .visualization import draw_detections, generate_colors

__all__ = [
    'non_max_suppression',
    'compute_iou',
    'calculate_metrics',
    'load_ground_truth',
    'draw_detections',
    'generate_colors',
]

