import cv2
import argparse
import os
import glob
import json
import csv
from datetime import datetime
from detector import VehicleDetector
from annotation_parser import AnnotationParser
from utils import calculate_detection_metrics, normalize_class_name


class NewFormatEvaluator:
    def __init__(self, images_path, annotation_file, detector, iou_threshold=0.5):
        self.images_path = images_path
        self.annotation_file = annotation_file
        self.detector = detector
        self.iou_threshold = iou_threshold

        # Парсим разметку
        self.annotation_parser = AnnotationParser(annotation_file, images_path)

        # Получаем список изображений
        self.image_files = self._get_sorted_image_files()

        print(f"Loaded {len(self.image_files)} images for evaluation")
        print(f"Found annotations for {self.annotation_parser.get_total_frames_with_annotations()} frames")
        print(f"Total objects in annotations: {self.annotation_parser.get_total_objects()}")

        # Диагностика
        self._diagnose_annotations()

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
            # Пробуем извлечь число из имени файла
            return int(''.join(filter(str.isdigit, filename)))
        except:
            return 0

    def _diagnose_annotations(self):
        """Диагностика разметки"""
        print("\n=== ANNOTATION DIAGNOSIS ===")

        # Проверяем первые 5 кадров с разметкой
        frame_numbers = self.annotation_parser.get_frame_numbers()[:5]
        for frame_num in frame_numbers:
            annotations = self.annotation_parser.get_ground_truth_for_frame(frame_num)
            print(f"Frame {frame_num}: {len(annotations)} objects")
            for ann in annotations[:3]:  # Показываем первые 3 объекта
                bbox = ann['bbox']
                print(f"  - {ann['class_name']}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

        # Проверяем соответствие изображений и разметки
        matched_frames = 0
        sample_matches = []

        for image_file in self.image_files[:20]:  # Проверяем первые 20 изображений
            frame_num = self._extract_frame_number(image_file)
            annotations = self.annotation_parser.get_ground_truth_for_frame(frame_num)
            if annotations:
                matched_frames += 1
                if len(sample_matches) < 3:
                    sample_matches.append((frame_num, len(annotations)))

        print(f"\nImage-annotation matching (first 20):")
        print(f"  Matched frames: {matched_frames}/20")
        if sample_matches:
            print(f"  Sample matches: {sample_matches}")

        if matched_frames == 0:
            print("\nWARNING: No matching frames found!")
            print("Possible issues:")
            print("  1. Frame numbers in annotation file don't match image file names")
            print("  2. Image files are named differently than expected")
            print("  3. Annotation file contains different frame range")

            # Показываем примеры имен файлов и номеров кадров
            print("\nFirst 5 image files:")
            for img_file in self.image_files[:5]:
                frame_num = self._extract_frame_number(img_file)
                print(f"  {os.path.basename(img_file)} -> Frame {frame_num}")

    def evaluate(self, output_dir="./evaluation_results_new", save_detections=False, max_images=None):
        """Оценка точности с новым форматом разметки"""
        os.makedirs(output_dir, exist_ok=True)

        all_metrics = []
        total_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_detections': 0,
            'total_ground_truth': 0
        }

        print("\nStarting evaluation with new format...")
        print("=" * 80)

        # Ограничиваем количество изображений для тестирования
        if max_images:
            image_files = self.image_files[:max_images]
        else:
            image_files = self.image_files

        processed_count = 0
        frames_with_annotations = 0
        frames_evaluated = 0

        for i, image_file in enumerate(image_files):
            frame_number = self._extract_frame_number(image_file)

            if i % 100 == 0:
                print(f"Processing image {i + 1}/{len(image_files)} (Frame {frame_number})...")

            # Загрузка изображения
            image = cv2.imread(image_file)
            if image is None:
                print(f"Error loading image: {image_file}")
                continue

            # Получаем размеры изображения для проверки координат
            image_height, image_width = image.shape[:2]

            # Детектирование
            detections = self.detector.detect_vehicles(image)
            vehicle_detections = [d for d in detections if
                                  d['class_name'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]

            # Загрузка ground truth для этого кадра
            ground_truth = self.annotation_parser.get_ground_truth_for_frame(frame_number, image_width, image_height)

            # Нормализуем названия классов в ground truth
            for gt in ground_truth:
                gt['class_name'] = normalize_class_name(gt['class_name'])

            # Вычисление метрик только если есть ground truth
            if ground_truth:
                frames_with_annotations += 1
                metrics = calculate_detection_metrics(vehicle_detections, ground_truth, self.iou_threshold)

                # Сохранение метрик для этого изображения
                frame_metrics = {
                    'frame_number': frame_number,
                    'image_file': os.path.basename(image_file),
                    'image_width': image_width,
                    'image_height': image_height,
                    **metrics
                }
                all_metrics.append(frame_metrics)

                # Агрегация статистики
                total_stats['true_positives'] += metrics['true_positives']
                total_stats['false_positives'] += metrics['false_positives']
                total_stats['false_negatives'] += metrics['false_negatives']
                total_stats['total_detections'] += metrics['total_detections']
                total_stats['total_ground_truth'] += metrics['total_ground_truth']

                # Вывод прогресса для кадров с аннотациями
                if frames_evaluated % 50 == 0:
                    print(
                        f"  Frame {frame_number}: {metrics['true_positives']} TP, {metrics['false_positives']} FP, {metrics['false_negatives']} FN")

                frames_evaluated += 1

                # Сохранение изображения с детекциями (опционально)
                if save_detections and frames_evaluated % 20 == 0:  # Сохраняем каждое 20-е изображение с аннотациями
                    self.save_detection_image(image, vehicle_detections, ground_truth, metrics, output_dir,
                                              frame_number)

            processed_count += 1

        # Вычисление общих метрик
        if frames_with_annotations > 0:
            overall_precision = total_stats['true_positives'] / (
                        total_stats['true_positives'] + total_stats['false_positives']) if (total_stats[
                                                                                                'true_positives'] +
                                                                                            total_stats[
                                                                                                'false_positives']) > 0 else 0
            overall_recall = total_stats['true_positives'] / (
                        total_stats['true_positives'] + total_stats['false_negatives']) if (total_stats[
                                                                                                'true_positives'] +
                                                                                            total_stats[
                                                                                                'false_negatives']) > 0 else 0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (
                                                                                                                        overall_precision + overall_recall) > 0 else 0
        else:
            overall_precision = overall_recall = overall_f1 = 0

        overall_metrics = {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1_score': overall_f1,
            'total_true_positives': total_stats['true_positives'],
            'total_false_positives': total_stats['false_positives'],
            'total_false_negatives': total_stats['false_negatives'],
            'total_detections': total_stats['total_detections'],
            'total_ground_truth': total_stats['total_ground_truth'],
            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'iou_threshold': self.iou_threshold,
            'total_images_processed': processed_count,
            'frames_with_annotations': frames_with_annotations,
            'frames_evaluated': frames_evaluated,
            'annotation_file': os.path.basename(self.annotation_file)
        }

        # Сохранение результатов
        self.save_results(all_metrics, overall_metrics, output_dir)

        # Вывод результатов
        self.print_results(overall_metrics)

        return overall_metrics

    def save_detection_image(self, image, detections, ground_truth, metrics, output_dir, frame_number):
        """Сохранение изображения с детекциями и метриками"""
        from utils import draw_detections_with_metrics

        result_image = draw_detections_with_metrics(image, detections, ground_truth, metrics)
        output_path = os.path.join(output_dir, f"frame_{frame_number:06d}_eval.jpg")
        cv2.imwrite(output_path, result_image)
        print(f"    Saved visualization: {os.path.basename(output_path)}")

    def save_results(self, all_metrics, overall_metrics, output_dir):
        """Сохранение результатов в файлы"""
        # Сохранение в JSON
        results = {
            'overall_metrics': overall_metrics,
            'per_image_metrics': all_metrics
        }

        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Сохранение в CSV (per image metrics)
        if all_metrics:
            csv_file = os.path.join(output_dir, 'per_image_metrics.csv')
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
                writer.writeheader()
                writer.writerows(all_metrics)

        # Сохранение сводки
        summary_file = os.path.join(output_dir, 'overall_metrics.csv')
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in overall_metrics.items():
                writer.writerow([key, value])

        print(f"Results saved to {output_dir}")

    def print_results(self, metrics):
        """Вывод результатов в консоль"""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS (NEW FORMAT)")
        print("=" * 80)
        print(f"Annotation File: {metrics['annotation_file']}")
        print(f"Total Images Processed: {metrics['total_images_processed']}")
        print(f"Frames with Annotations: {metrics['frames_with_annotations']}")
        print(f"Frames Evaluated: {metrics['frames_evaluated']}")
        print(f"Total Ground Truth Objects: {metrics['total_ground_truth']}")
        print(f"Total Detections: {metrics['total_detections']}")
        print(f"IOU Threshold: {metrics['iou_threshold']}")

        if metrics['frames_with_annotations'] > 0:
            print("\n--- Detection Metrics ---")
            print(
                f"Precision: {metrics['overall_precision']:.4f} ({metrics['total_true_positives']}/{metrics['total_true_positives'] + metrics['total_false_positives']})")
            print(
                f"Recall:    {metrics['overall_recall']:.4f} ({metrics['total_true_positives']}/{metrics['total_true_positives'] + metrics['total_false_negatives']})")
            print(f"F1-Score:  {metrics['overall_f1_score']:.4f}")
            print("\n--- Detection Statistics ---")
            print(f"True Positives:  {metrics['total_true_positives']}")
            print(f"False Positives: {metrics['total_false_positives']}")
            print(f"False Negatives: {metrics['total_false_negatives']}")

            # Дополнительная статистика
            detection_rate = metrics['total_detections'] / metrics['frames_evaluated'] if metrics[
                                                                                              'frames_evaluated'] > 0 else 0
            gt_per_frame = metrics['total_ground_truth'] / metrics['frames_evaluated'] if metrics[
                                                                                              'frames_evaluated'] > 0 else 0
            print(f"\n--- Per Frame Statistics ---")
            print(f"Average detections per frame: {detection_rate:.2f}")
            print(f"Average ground truth per frame: {gt_per_frame:.2f}")
        else:
            print("\nWARNING: No frames with annotations found!")
            print("Please check:")
            print("  1. Annotation file path is correct")
            print("  2. Frame numbers in annotation file match image file names")
            print("  3. Annotation file format is: [frame] [class] [x1] [y1] [x2] [y2]")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate detection accuracy with new annotation format')
    parser.add_argument('--images_path', type=str, required=True, help='Path to images directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotation text file')
    parser.add_argument('--model_type', type=str, default='yolo', choices=['yolo', 'ssd'])
    parser.add_argument('--model_path', type=str, default='models_data/yolov4-tiny.weights')
    parser.add_argument('--config_path', type=str, default='models_data/yolov4-tiny.cfg')
    parser.add_argument('--classes_path', type=str, default='models_data/coco.names')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IOU threshold for matching')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results_new',
                        help='Output directory for results')
    parser.add_argument('--save_detections', action='store_true', help='Save detection images')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')

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

    # Создаем оценщика для нового формата
    evaluator = NewFormatEvaluator(
        images_path=args.images_path,
        annotation_file=args.annotation_file,
        detector=detector,
        iou_threshold=args.iou_threshold
    )

    # Запускаем оценку
    results = evaluator.evaluate(
        output_dir=args.output_dir,
        save_detections=args.save_detections,
        max_images=args.max_images
    )

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
