import cv2
import argparse
import os
import glob
from detector import VehicleDetector
from utils import load_ground_truth, draw_detections, download_models, download_dataset


def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection using OpenCV DNN')
    parser.add_argument('--images_path', type=str, default='./dataset/images', help='Path to images directory')
    parser.add_argument('--annotations_path', type=str, default='./dataset/annotations',
                        help='Path to annotations directory')
    parser.add_argument('--model_type', type=str, default='yolo', choices=['yolo', 'ssd'], help='Model type')
    parser.add_argument('--model_path', type=str, help='Path to model weights')
    parser.add_argument('--config_path', type=str, help='Path to model configuration')
    parser.add_argument('--classes_path', type=str, help='Path to classes file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--show_detections', action='store_true', help='Show detection results')
    parser.add_argument('--download_models', action='store_true', help='Download pre-trained models')
    parser.add_argument('--download_dataset', action='store_true', help='Download dataset with annotations')

    args = parser.parse_args()

    # Загрузка датасета при необходимости
    if args.download_dataset:
        print("Downloading dataset...")
        images_url = "https://cloud.unn.ru/s/nLkk7BXBqapNgcE/download"
        annotations_url = "https://cloud.unn.ru/s/j4wA4nx8mZ4yfqD/download"
        download_dataset(images_url, annotations_url)
        return

    # Загрузка моделей при необходимости
    if args.download_models:
        print("Downloading pre-trained models...")
        download_models()
        print("Models downloaded successfully!")
        return

    # Пути по умолчанию для моделей
    if args.model_type == 'yolo' and not all([args.model_path, args.config_path, args.classes_path]):
        args.model_path = 'models_data/yolov4-tiny.weights'
        args.config_path = 'models_data/yolov4-tiny.cfg'
        args.classes_path = 'models_data/coco.names'
        print(f"Using default YOLO model: {args.model_path}")

    # Проверка существования файлов моделей
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        print("Use --download_models flag to download pre-trained models")
        return

    if args.config_path and not os.path.exists(args.config_path):
        print(f"Error: Config path '{args.config_path}' does not exist")
        print("Use --download_models flag to download pre-trained models")
        return

    if not os.path.exists(args.classes_path):
        print(f"Error: Classes path '{args.classes_path}' does not exist")
        print("Use --download_models flag to download pre-trained models")
        return

    # Проверка существования пути с изображениями
    if not os.path.exists(args.images_path):
        print(f"Error: Images path '{args.images_path}' does not exist")
        print("Use --download_dataset flag to download the dataset")
        return

    # Проверка существования пути с разметкой
    if not os.path.exists(args.annotations_path):
        print(f"Warning: Annotations path '{args.annotations_path}' does not exist")
        print("Metrics calculation will be limited")

    # Создание детектора
    try:
        detector = VehicleDetector(
            model_type=args.model_type,
            model_path=args.model_path,
            config_path=args.config_path,
            classes_file=args.classes_path
        )
        detector.set_confidence_threshold(args.confidence)
        print(f"Detector initialized successfully with {args.model_type.upper()} model")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # Получение списка изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.images_path, ext)))

    if not image_files:
        print(f"No images found in '{args.images_path}'")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return

    print(f"Found {len(image_files)} images")

    # Обработка изображений
    total_tpr = 0
    total_fdr = 0
    processed_count = 0

    for image_file in image_files:
        # Загрузка изображения
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error loading image: {image_file}")
            continue

        print(f"\nProcessing: {os.path.basename(image_file)}")

        # Загрузка ground truth
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        annotation_file = os.path.join(args.annotations_path, f"{image_name}.txt")
        ground_truth = load_ground_truth(annotation_file, image.shape[1], image.shape[0])

        # Детектирование
        try:
            detections = detector.detect_vehicles(image)

            # Фильтрация только транспортных средств
            vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
            vehicle_detections = [d for d in detections if d['class_name'] in vehicle_classes]

            # Вычисление метрик (только если есть ground truth)
            if ground_truth:
                tpr, fdr = detector.calculate_metrics(vehicle_detections, ground_truth)

                total_tpr += tpr
                total_fdr += fdr
                processed_count += 1

                print(f"  Detections: {len(vehicle_detections)}, Ground truth: {len(ground_truth)}")
                print(f"  TPR: {tpr:.3f}, FDR: {fdr:.3f}")
            else:
                print(f"  Detections: {len(vehicle_detections)}")
                print("  No ground truth available for metrics calculation")

            # Отображение результатов
            if args.show_detections:
                result_image = draw_detections(image.copy(), vehicle_detections, ground_truth)

                # Добавление информации о метриках
                if ground_truth:
                    cv2.putText(result_image, f"TPR: {tpr:.3f} FDR: {fdr:.3f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_image, f"Detections: {len(vehicle_detections)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if ground_truth:
                    cv2.putText(result_image, f"Ground truth: {len(ground_truth)}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Легенда
                cv2.putText(result_image, "Green: Detections",
                            (10, result_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(result_image, "Blue: Ground Truth",
                            (10, result_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Изменение размера для отображения, если изображение слишком большое
                height, width = result_image.shape[:2]
                if width > 1200 or height > 800:
                    scale = min(1200 / width, 800 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    result_image = cv2.resize(result_image, (new_width, new_height))

                cv2.imshow('Vehicle Detection', result_image)

                # Выход по нажатию 'q', продолжение по любой другой клавише
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Сохранение результата
                    output_path = os.path.splitext(image_file)[0] + '_result.jpg'
                    cv2.imwrite(output_path, result_image)
                    print(f"Result saved to {output_path}")

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

    # Вывод средних метрик
    if processed_count > 0:
        avg_tpr = total_tpr / processed_count
        avg_fdr = total_fdr / processed_count
        print(f"\n{'=' * 50}")
        print(f"Average metrics over {processed_count} images:")
        print(f"TPR (True Positive Rate): {avg_tpr:.3f}")
        print(f"FDR (False Discovery Rate): {avg_fdr:.3f}")
        print(f"{'=' * 50}")
    else:
        print("No images with ground truth were successfully processed")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()