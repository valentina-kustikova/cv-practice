"""
Демонстрационное приложение для детектирования транспортных средств.

Практическая работа №2. Детектирование объектов на изображениях 
с использованием библиотеки OpenCV.

Использование:
    python main.py --data_path ./data --labels_path labels.txt --model yolov8 
                   --model_path models/yolov8l.onnx --show
"""

import argparse
import os
import sys
import glob
import time
import random
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from detectors import Detection, create_detector, BaseDetector


@dataclass
class GroundTruthBox:
    """Класс для хранения ground truth разметки."""
    frame_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int


def load_ground_truth(labels_path: str) -> Dict[int, List[GroundTruthBox]]:
    """
    Загрузка ground truth разметки из файла.
    
    Args:
        labels_path: путь к файлу разметки
        
    Returns:
        gt_dict: словарь {frame_id: [GroundTruthBox, ...]}
    """
    gt_dict = {}
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                frame_id = int(parts[0])
                class_name = parts[1]
                x1, y1, x2, y2 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                
                gt_box = GroundTruthBox(frame_id, class_name, x1, y1, x2, y2)
                
                if frame_id not in gt_dict:
                    gt_dict[frame_id] = []
                gt_dict[frame_id].append(gt_box)
    
    return gt_dict


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Вычисление Intersection over Union (IoU) для двух bounding boxes.
    
    Args:
        box1, box2: (x1, y1, x2, y2)
        
    Returns:
        iou: значение IoU [0, 1]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_metrics(detections: List[Detection], 
                      ground_truth: List[GroundTruthBox],
                      iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Вычисление TP, FP, FN для одного кадра.
    
    Args:
        detections: список детекций
        ground_truth: список ground truth boxes
        iou_threshold: порог IoU для считания детекции правильной
        
    Returns:
        tp, fp, fn: True Positives, False Positives, False Negatives
    """
    tp = 0
    fp = 0
    
    gt_matched = [False] * len(ground_truth)
    
    # Сортируем детекции по confidence в порядке убывания,
    # чтобы лучшие предсказания обрабатывались первыми
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    for det in sorted_detections:
        det_box = (det.x1, det.y1, det.x2, det.y2)
        det_class = det.class_name.upper()
        
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truth):
            if gt_matched[i]:
                continue
            
            gt_box = (gt.x1, gt.y1, gt.x2, gt.y2)
            gt_class = gt.class_name.upper()
            
            # Проверяем совпадение классов
            if det_class != gt_class:
                continue
            
            iou = calculate_iou(det_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    # FN = ground truth boxes, которые не были найдены
    fn = sum(1 for matched in gt_matched if not matched)
    
    return tp, fp, fn


def calculate_tpr_fdr(total_tp: int, total_fp: int, total_fn: int) -> Tuple[float, float]:
    """
    Вычисление TPR (True Positive Rate) и FDR (False Discovery Rate).
    
    TPR = TP / (TP + FN) - доля правильно найденных объектов
    FDR = FP / (TP + FP) - доля ложных срабатываний среди всех детекций
    
    Args:
        total_tp: общее количество True Positives
        total_fp: общее количество False Positives
        total_fn: общее количество False Negatives
        
    Returns:
        tpr, fdr: метрики качества
    """
    tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    fdr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    return tpr, fdr


# Цвета для разных классов (BGR)
CLASS_COLORS = {
    'CAR': (0, 255, 0),      # Зелёный
    'BUS': (255, 0, 0),      # Синий
}


def draw_detections(image: np.ndarray, 
                    detections: List[Detection],
                    ground_truth: Optional[List[GroundTruthBox]] = None) -> np.ndarray:
    """
    Отрисовка детекций и ground truth на изображении.
    
    Требования:
    - Прямоугольники разных цветов для разных классов
    - В левом верхнем углу прямоугольника: название класса и confidence (3 знака)
    - Над прямоугольником: наблюдаемый класс (ground truth)
    
    Args:
        image: исходное изображение
        detections: список детекций
        ground_truth: список ground truth boxes (опционально)
        
    Returns:
        image: изображение с отрисованными детекциями
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Отрисовка ground truth (пунктирные линии)
    if ground_truth:
        for gt in ground_truth:
            color = CLASS_COLORS.get(gt.class_name.upper(), (128, 128, 128))
            # Рисуем пунктирную рамку для ground truth
            draw_dashed_rectangle(img, (gt.x1, gt.y1), (gt.x2, gt.y2), color, 2)
            
            # Над прямоугольником пишем наблюдаемый класс
            label = f"GT: {gt.class_name}"
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.putText(img, label, (gt.x1, gt.y1 - 5), font, font_scale, color, thickness)
    
    # Отрисовка детекций (сплошные линии)
    for det in detections:
        color = CLASS_COLORS.get(det.class_name.upper(), (0, 255, 255))
        
        # Рисуем прямоугольник
        cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        
        # В левом верхнем углу: класс и confidence
        label = f"{det.class_name}: {det.confidence:.3f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Фон для текста
        cv2.rectangle(img, 
                      (det.x1, det.y1), 
                      (det.x1 + label_w, det.y1 + label_h + baseline), 
                      color, -1)
        cv2.putText(img, label, (det.x1, det.y1 + label_h), 
                    font, font_scale, (0, 0, 0), thickness)
    
    return img


def draw_dashed_rectangle(img: np.ndarray, 
                          pt1: Tuple[int, int], 
                          pt2: Tuple[int, int], 
                          color: Tuple[int, int, int], 
                          thickness: int = 1,
                          dash_length: int = 10) -> None:
    """Отрисовка пунктирного прямоугольника."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Верхняя и нижняя линии
    for x in range(x1, x2, dash_length * 2):
        cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    
    # Левая и правая линии
    for y in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def get_image_files(data_path: str) -> List[str]:
    """
    Получение списка файлов изображений из директории.
    
    Args:
        data_path: путь к директории с изображениями
        
    Returns:
        files: отсортированный список путей к файлам
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_path, ext)))
    
    return sorted(files)


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Детектирование транспортных средств с использованием OpenCV DNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python main.py --data_path ./data --labels_path labels.txt --model yolov8 --model_path models/yolov8l.onnx --show
  python main.py --data_path ./data --labels_path labels.txt --model ssd --model_path models/ssd.pb --config_path models/ssd.pbtxt
  python main.py --data_path ./data --labels_path labels.txt --model nanodet --model_path models/nanodet-plus.onnx

Доступные модели:
  yolov8    - YOLOv8l (ONNX)
  ssd       - SSD Inception v2 (TensorFlow)
  nanodet   - NanoDet-Plus (ONNX)
        '''
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Путь к директории с изображениями')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Путь к файлу разметки (ground truth)')
    parser.add_argument('--model', type=str, required=True,
                        choices=['yolov8', 'ssd', 'nanodet'],
                        help='Название модели для детекции')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к файлу модели')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Путь к файлу конфигурации (для SSD)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Порог уверенности для детекций (по умолчанию: 0.5)')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='Порог NMS (по умолчанию: 0.4)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='Порог IoU для вычисления метрик (по умолчанию: 0.5)')
    parser.add_argument('--show', action='store_true',
                        help='Отображать кадры с детекциями')
    parser.add_argument('--save_output', type=str, default=None,
                        help='Путь для сохранения результатов (опционально)')
    parser.add_argument('--save_fp', type=str, default=None,
                        help='Путь для сохранения кадров с FP (ложными срабатываниями)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Максимальное количество кадров для обработки')
    parser.add_argument('--random', action='store_true',
                        help='Выбирать случайные кадры из датасета')
    
    return parser.parse_args()


def main():
    """Главная функция приложения."""
    args = parse_args()
    
    # Проверка существования путей
    if not os.path.exists(args.data_path):
        print(f"Ошибка: директория с данными не найдена: {args.data_path}")
        sys.exit(1)
    
    if not os.path.exists(args.labels_path):
        print(f"Ошибка: файл разметки не найден: {args.labels_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Ошибка: файл модели не найден: {args.model_path}")
        sys.exit(1)
    
    if args.model == 'ssd' and (args.config_path is None or not os.path.exists(args.config_path)):
        print(f"Ошибка: для SSD требуется файл конфигурации (--config_path)")
        sys.exit(1)
    
    # Создание директории для сохранения результатов
    if args.save_output:
        os.makedirs(args.save_output, exist_ok=True)
    
    # Создание директории для кадров с FP
    if args.save_fp:
        os.makedirs(args.save_fp, exist_ok=True)
    
    # Загрузка ground truth
    print("Загрузка разметки...")
    ground_truth = load_ground_truth(args.labels_path)
    print(f"Загружено {sum(len(v) for v in ground_truth.values())} объектов для {len(ground_truth)} кадров")
    
    # Создание детектора
    print(f"\nИнициализация детектора {args.model}...")
    try:
        detector = create_detector(
            model_name=args.model,
            model_path=args.model_path,
            config_path=args.config_path,
            confidence_threshold=args.confidence,
            nms_threshold=args.nms_threshold
        )
        print(f"Детектор {detector.name} успешно загружен")
    except Exception as e:
        print(f"Ошибка при загрузке детектора: {e}")
        sys.exit(1)
    
    # Получение списка изображений
    image_files = get_image_files(args.data_path)
    
    # Выбор случайных кадров если указано
    if args.random and args.max_frames and len(image_files) > args.max_frames:
        random.seed(42)  # Для воспроизводимости
        image_files = random.sample(image_files, args.max_frames)
        image_files = sorted(image_files)
    elif args.max_frames:
        image_files = image_files[:args.max_frames]
    
    print(f"\nНайдено {len(image_files)} изображений")
    
    # Счётчики для метрик
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Для измерения времени
    inference_times = []
    
    print("\nНачало обработки...")
    print("-" * 60)
    
    for i, image_path in enumerate(image_files):
        # Получаем frame_id из имени файла
        filename = os.path.basename(image_path)
        frame_id = int(os.path.splitext(filename)[0])
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"Предупреждение: не удалось загрузить {image_path}")
            continue
        
        # Детекция с замером времени
        start_time = time.perf_counter()
        detections = detector.detect(image)
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        
        # Получение ground truth для текущего кадра
        gt_boxes = ground_truth.get(frame_id, [])
        
        # Вычисление метрик для текущего кадра
        tp, fp, fn = calculate_metrics(detections, gt_boxes, args.iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Сохранение кадров с FP (ложными срабатываниями)
        if args.save_fp and fp > 0:
            vis_image = draw_detections(image, detections, gt_boxes)
            info_text = f"Frame: {frame_id} | TP: {tp}, FP: {fp}, FN: {fn}"
            cv2.putText(vis_image, info_text, (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            output_path = os.path.join(args.save_fp, f"{frame_id:06d}_fp{fp}.jpg")
            cv2.imwrite(output_path, vis_image)
        
        # Прогресс
        if (i + 1) % 100 == 0 or i == len(image_files) - 1:
            tpr, fdr = calculate_tpr_fdr(total_tp, total_fp, total_fn)
            print(f"Обработано: {i + 1}/{len(image_files)} | "
                  f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn} | "
                  f"TPR: {tpr:.4f}, FDR: {fdr:.4f}")
        
        # Отображение
        if args.show:
            vis_image = draw_detections(image, detections, gt_boxes)
            
            # Добавляем информацию на изображение
            info_text = f"Frame: {frame_id} | Detections: {len(detections)} | GT: {len(gt_boxes)}"
            cv2.putText(vis_image, info_text, (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f"Detection - {detector.name}", vis_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nОстановлено пользователем")
                break
            elif key == ord(' '):
                # Пауза
                cv2.waitKey(0)
        
        # Сохранение результатов
        if args.save_output:
            vis_image = draw_detections(image, detections, gt_boxes)
            output_path = os.path.join(args.save_output, filename)
            cv2.imwrite(output_path, vis_image)
    
    if args.show:
        cv2.destroyAllWindows()
    
    # Итоговые метрики
    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    tpr, fdr = calculate_tpr_fdr(total_tp, total_fp, total_fn)
    
    # Статистика времени
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_time_ms = avg_time * 1000
    
    print(f"Модель: {detector.name}")
    print(f"Обработано кадров: {min(i + 1, len(image_files))}")
    print(f"\nМетрики детекции:")
    print(f"  True Positives (TP):  {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print(f"\nПоказатели качества:")
    print(f"  TPR (True Positive Rate):   {tpr:.4f} ({tpr * 100:.2f}%)")
    print(f"  FDR (False Discovery Rate): {fdr:.4f} ({fdr * 100:.2f}%)")
    print(f"\nПроизводительность:")
    print(f"  Среднее время на кадр: {avg_time_ms:.2f} мс")
    print(f"  FPS: {1/avg_time:.2f}" if avg_time > 0 else "  FPS: N/A")
    print("=" * 60)


if __name__ == "__main__":
    main()
