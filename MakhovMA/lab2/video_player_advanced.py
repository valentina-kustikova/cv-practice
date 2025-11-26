import cv2
import os
import glob
import time
import numpy as np
from detector import VehicleDetector
from annotation_parser import AnnotationParser
from utils import calculate_detection_metrics, normalize_class_name, calculate_iou


class AdvancedVideoPlayer:
    def __init__(self, images_path, annotation_file, detector, delay=30):
        self.images_path = images_path
        self.annotation_file = annotation_file
        self.detector = detector
        self.delay = delay

        # Парсим разметку
        self.annotation_parser = AnnotationParser(annotation_file, images_path)

        # Получаем список изображений
        self.image_files = self._get_sorted_image_files()
        self.total_frames = len(self.image_files)

        # Статистика
        self.cumulative_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_detections': 0,
            'total_ground_truth': 0,
            'frames_with_detections': 0,
            'frames_evaluated': 0
        }

        # Настройки отображения
        self.colors = {
            'true_positive': (0, 255, 0),  # Зеленый - правильные детекции
            'false_positive': (0, 165, 255),  # Оранжевый - ложные срабатывания
            'false_negative': (0, 0, 255),  # Красный - пропущенные объекты
            'ground_truth': (255, 0, 0),  # Синий - разметка
            'text': (255, 255, 255),  # Белый - текст
            'background': (0, 0, 0)  # Черный - фон
        }

        print(f"Loaded {self.total_frames} frames")
        print(f"Found annotations for {self.annotation_parser.get_total_frames_with_annotations()} frames")

    def _get_sorted_image_files(self):
        """Получение и сортировка списка изображений"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_path, ext)))

        # Сортируем по номеру кадра из имени файла
        image_files.sort(key=lambda x: self._extract_frame_number(x))
        return image_files

    def _extract_frame_number(self, image_path):
        """Извлечение номера кадра из имени файла"""
        filename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            return int(''.join(filter(str.isdigit, filename)))
        except:
            return 0

    def _classify_detections(self, detections, ground_truth, iou_threshold=0.5):
        """Классификация детекций на TP, FP, FN"""
        true_positives = []
        false_positives = []
        false_negatives = list(ground_truth)  # Изначально все GT считаем пропущенными

        matched_gt = set()
        matched_detections = set()

        # Сопоставление детекций с ground truth
        for det_idx, detection in enumerate(detections):
            matched = False
            det_bbox = detection['bbox']
            det_class = normalize_class_name(detection['class_name'])

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                gt_bbox = gt['bbox']
                gt_class = normalize_class_name(gt['class_name'])

                # Проверка совпадения классов
                if det_class != gt_class:
                    continue

                iou = calculate_iou(det_bbox, gt_bbox)

                if iou >= iou_threshold:
                    # True Positive
                    true_positives.append({
                        **detection,
                        'matched_gt': gt,
                        'iou': iou
                    })
                    matched_gt.add(gt_idx)
                    if gt in false_negatives:
                        false_negatives.remove(gt)
                    matched = True
                    break

            if not matched:
                # False Positive
                false_positives.append(detection)

        return true_positives, false_positives, false_negatives

    def draw_detections_with_analysis(self, image, true_positives, false_positives, false_negatives, ground_truth,
                                      metrics, frame_number):
        """Отрисовка детекций с цветовой кодировкой по типу"""
        result_image = image.copy()
        height, width = result_image.shape[:2]

        # Отрисовка ground truth (синим)
        for gt in ground_truth:
            x1, y1, x2, y2 = gt['bbox']
            class_name = gt['class_name']

            # Ground truth bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), self.colors['ground_truth'], 2)

            # Подпись ground truth
            label = f"GT: {class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), self.colors['ground_truth'], -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)

        # Отрисовка False Positives (оранжевый)
        for fp in false_positives:
            x1, y1, x2, y2 = fp['bbox']
            class_name = fp['class_name']
            confidence = fp['confidence']

            cv2.rectangle(result_image, (x1, y1), (x2, y2), self.colors['false_positive'], 2)
            label = f"FP: {class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), self.colors['false_positive'], -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)

        # Отрисовка True Positives (зеленый)
        for tp in true_positives:
            x1, y1, x2, y2 = tp['bbox']
            class_name = tp['class_name']
            confidence = tp['confidence']
            iou = tp.get('iou', 0)

            cv2.rectangle(result_image, (x1, y1), (x2, y2), self.colors['true_positive'], 3)
            label = f"TP: {class_name} ({confidence:.2f}, IoU:{iou:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), self.colors['true_positive'], -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)

        # Отрисовка False Negatives (красный)
        for fn in false_negatives:
            x1, y1, x2, y2 = fn['bbox']
            class_name = fn['class_name']

            cv2.rectangle(result_image, (x1, y1), (x2, y2), self.colors['false_negative'], 2)
            # Пунктирная линия для FN
            dash_length = 10
            for x in range(x1, x2, dash_length * 2):
                cv2.line(result_image, (x, y1), (min(x + dash_length, x2), y1), self.colors['false_negative'], 2)
                cv2.line(result_image, (x, y2), (min(x + dash_length, x2), y2), self.colors['false_negative'], 2)
            for y in range(y1, y2, dash_length * 2):
                cv2.line(result_image, (x1, y), (x1, min(y + dash_length, y2)), self.colors['false_negative'], 2)
                cv2.line(result_image, (x2, y), (x2, min(y + dash_length, y2)), self.colors['false_negative'], 2)

            label = f"FN: {class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), self.colors['false_negative'], -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)

        # Панель статистики
        stats_panel_height = 200
        stats_panel = np.zeros((stats_panel_height, width, 3), dtype=np.uint8)

        # Вычисление кумулятивных метрик
        cum_tp = self.cumulative_stats['true_positives']
        cum_fp = self.cumulative_stats['false_positives']
        cum_fn = self.cumulative_stats['false_negatives']

        cum_precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0
        cum_recall = cum_tp / (cum_tp + cum_fn) if (cum_tp + cum_fn) > 0 else 0

        # Текущие метрики кадра
        stats_text = [
            f"Frame: {frame_number}/{self.total_frames}",
            f"TP: {len(true_positives)}  FP: {len(false_positives)}  FN: {len(false_negatives)}",
            f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}",
            f"F1-Score: {metrics['f1_score']:.3f}",
            "",
            "Cumulative Stats:",
            f"Total TP: {cum_tp}",
            f"Total FP: {cum_fp}",
            f"Total FN: {cum_fn}",
            f"Precision: {cum_precision:.3f}",
            f"Recall: {cum_recall:.3f}"
        ]

        # Отрисовка текста статистики
        for i, text in enumerate(stats_text):
            y_position = 30 + i * 25
            if y_position < stats_panel_height - 10:  # Проверяем чтобы не выйти за границы
                cv2.putText(stats_panel, text, (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

        # Легенда
        legend_y = 150
        legend_items = [
            ("True Positive (Green)", self.colors['true_positive']),
            ("False Positive (Orange)", self.colors['false_positive']),
            ("False Negative (Red)", self.colors['false_negative']),
            ("Ground Truth (Blue)", self.colors['ground_truth'])
        ]

        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + i * 20
            if y_pos < stats_panel_height - 10:
                # Цветной квадратик
                cv2.rectangle(stats_panel, (width - 270, y_pos - 15),
                              (width - 250, y_pos - 5), color, -1)
                # Текст
                cv2.putText(stats_panel, text, (width - 240, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        # Объединяем изображение с панелью статистики
        result_image = np.vstack([result_image, stats_panel])

        return result_image

    def run(self, start_frame=0):
        """Запуск улучшенного видеоплеера"""
        if self.total_frames == 0:
            print("No images found!")
            return

        self.current_frame = start_frame
        self.paused = False
        self.speed = 1.0

        print("\n=== ADVANCED VIDEO PLAYER ===")
        print("Controls:")
        print("  SPACE - Play/Pause")
        print("  → - Next frame")
        print("  ← - Previous frame")
        print("  S - Save current frame")
        print("  Q - Quit")
        print("  R - Reset to first frame")
        print("  + - Increase speed")
        print("  - - Decrease speed")
        print("  C - Clear cumulative stats")
        print("=" * 50)

        while self.current_frame < self.total_frames:
            if not self.paused:
                self.process_current_frame()
                self.current_frame += 1

                actual_delay = max(1, int(self.delay / self.speed))
                key = cv2.waitKey(actual_delay) & 0xFF
            else:
                key = cv2.waitKey(100) & 0xFF

            if self.handle_keypress(key):
                break

            self.update_window_title()

        cv2.destroyAllWindows()

    def process_current_frame(self):
        """Обработка текущего кадра"""
        if self.current_frame >= self.total_frames:
            return

        image_file = self.image_files[self.current_frame]
        frame_number = self._extract_frame_number(image_file)

        image = cv2.imread(image_file)
        if image is None:
            print(f"Error loading frame {self.current_frame}: {image_file}")
            return

        # Детектирование
        detections = self.detector.detect_vehicles(image)
        vehicle_detections = [d for d in detections if
                              d['class_name'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]

        # Загрузка ground truth
        ground_truth = self.annotation_parser.get_ground_truth_for_frame(frame_number, image.shape[1], image.shape[0])

        # Нормализация классов
        for gt in ground_truth:
            gt['class_name'] = normalize_class_name(gt['class_name'])

        # Классификация детекций
        true_positives, false_positives, false_negatives = self._classify_detections(vehicle_detections, ground_truth)

        # Вычисление метрик для текущего кадра
        if ground_truth:
            metrics = calculate_detection_metrics(vehicle_detections, ground_truth)

            # Обновление кумулятивной статистики
            self.cumulative_stats['true_positives'] += len(true_positives)
            self.cumulative_stats['false_positives'] += len(false_positives)
            self.cumulative_stats['false_negatives'] += len(false_negatives)
            self.cumulative_stats['total_detections'] += len(vehicle_detections)
            self.cumulative_stats['total_ground_truth'] += len(ground_truth)
            self.cumulative_stats['frames_evaluated'] += 1
        else:
            metrics = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }

        # Отрисовка
        result_image = self.draw_detections_with_analysis(
            image, true_positives, false_positives, false_negatives,
            ground_truth, metrics, frame_number
        )

        # Изменение размера если нужно
        height, width = result_image.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = 1200
            new_height = int(height * scale)
            result_image = cv2.resize(result_image, (new_width, new_height))

        cv2.imshow('Advanced Vehicle Detection', result_image)

        # Вывод в консоль каждые 50 кадров
        if self.current_frame % 50 == 0 and ground_truth:
            print(
                f"Frame {frame_number}: {len(true_positives)} TP, {len(false_positives)} FP, {len(false_negatives)} FN")

    def handle_keypress(self, key):
        """Обработка нажатий клавиш"""
        if key == ord('q') or key == 27:  # Q или ESC
            return True
        elif key == ord(' '):  # SPACE - пауза/продолжить
            self.paused = not self.paused
            print("Paused" if self.paused else "Playing")
        elif key == 83 or key == 3:  # Стрелка вправо
            self.paused = True
            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            self.process_current_frame()
        elif key == 81 or key == 2:  # Стрелка влево
            self.paused = True
            self.current_frame = max(self.current_frame - 1, 0)
            self.process_current_frame()
        elif key == ord('s'):  # S - сохранить
            self.save_current_frame()
        elif key == ord('r'):  # R - сброс
            self.current_frame = 0
            print("Reset to first frame")
        elif key == ord('+'):  # + - увеличить скорость
            self.speed = min(self.speed + 0.5, 5.0)
            print(f"Speed: {self.speed:.1f}x")
        elif key == ord('-'):  # - - уменьшить скорость
            self.speed = max(self.speed - 0.5, 0.5)
            print(f"Speed: {self.speed:.1f}x")
        elif key == ord('c'):  # C - очистить статистику
            self.cumulative_stats = {k: 0 for k in self.cumulative_stats}
            print("Cumulative stats cleared")

        return False

    def save_current_frame(self):
        """Сохранение текущего кадра"""
        if self.current_frame < self.total_frames:
            image_file = self.image_files[self.current_frame]
            frame_number = self._extract_frame_number(image_file)
            output_path = f"frame_{frame_number:06d}_analysis.jpg"

            # Перерисовываем и сохраняем
            image = cv2.imread(image_file)
            detections = self.detector.detect_vehicles(image)
            vehicle_detections = [d for d in detections if
                                  d['class_name'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]

            ground_truth = self.annotation_parser.get_ground_truth_for_frame(frame_number, image.shape[1],
                                                                             image.shape[0])
            for gt in ground_truth:
                gt['class_name'] = normalize_class_name(gt['class_name'])

            true_positives, false_positives, false_negatives = self._classify_detections(vehicle_detections,
                                                                                         ground_truth)
            metrics = calculate_detection_metrics(vehicle_detections, ground_truth) if ground_truth else {
                'precision': 0, 'recall': 0, 'f1_score': 0}

            result_image = self.draw_detections_with_analysis(
                image, true_positives, false_positives, false_negatives,
                ground_truth, metrics, frame_number
            )

            cv2.imwrite(output_path, result_image)
            print(f"Frame saved to: {output_path}")

    def update_window_title(self):
        """Обновление заголовка окна"""
        status = "PAUSED" if self.paused else "PLAYING"
        title = f"Advanced Vehicle Detection - {status} - Frame {self.current_frame + 1}/{self.total_frames} - {self.speed:.1f}x"
        cv2.setWindowTitle('Advanced Vehicle Detection', title)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Advanced Video Player for Vehicle Detection with Analysis')
    parser.add_argument('--images_path', type=str, required=True, help='Path to images directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotation text file')
    parser.add_argument('--model_type', type=str, default='yolo', choices=['yolo', 'ssd'])
    parser.add_argument('--model_path', type=str, default='models_data/yolov4-tiny.weights')
    parser.add_argument('--config_path', type=str, default='models_data/yolov4-tiny.cfg')
    parser.add_argument('--classes_path', type=str, default='models_data/coco.names')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--delay', type=int, default=30, help='Delay between frames in ms')
    parser.add_argument('--start_frame', type=int, default=0, help='Start from frame number')

    args = parser.parse_args()

    # Проверка путей
    if not os.path.exists(args.images_path):
        print(f"Error: Images path '{args.images_path}' does not exist")
        return

    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file '{args.annotation_file}' does not exist")
        return

    # Создаем детектор
    detector = VehicleDetector(
        model_type=args.model_type,
        model_path=args.model_path,
        config_path=args.config_path,
        classes_file=args.classes_path
    )
    detector.set_confidence_threshold(args.confidence)

    # Запускаем улучшенный видеоплеер
    player = AdvancedVideoPlayer(
        images_path=args.images_path,
        annotation_file=args.annotation_file,
        detector=detector,
        delay=args.delay
    )

    try:
        player.run(start_frame=args.start_frame)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    