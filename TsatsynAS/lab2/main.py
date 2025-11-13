import cv2
import os
import argparse
import glob
from detectors.ssd_detector import SSDDetector
from detectors.faster_rcnn_detector import FasterRCNNDetector
from detectors.yolov_detector import YOLODetector
from utils.metrics import calculate_metrics
import config


def load_ground_truth(annotation_path, current_frame_number, image_width, image_height):
    """
    Load ground truth annotations from mov03478.txt format for specific frame
    Format: "frame_number CLASS x1 y1 x2 y2"
    Example: "0 CAR 339 82 446 169"
    """
    ground_truth = []

    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    frame_num = int(parts[0])

                    # Load only annotations for current frame
                    if frame_num == current_frame_number:
                        class_name = parts[1].upper()  # CAR, BUS, TRUCK и т.д.

                        # Расширенное преобразование классов
                        class_mapping = {
                            'CAR': 'car',
                            'BUS': 'bus',
                            'TRUCK': 'truck',
                            'MOTORCYCLE': 'motorcycle',
                            'BICYCLE': 'bicycle',
                            'MOTORBIKE': 'motorcycle',
                            'VAN': 'car',  # часто ван считается машиной
                            'VEHICLE': 'car'  # общий класс
                        }

                        # Применяем маппинг или оставляем как есть (в нижнем регистре)
                        normalized_class = class_mapping.get(class_name, class_name.lower())

                        x1 = int(parts[2])
                        y1 = int(parts[3])
                        x2 = int(parts[4])
                        y2 = int(parts[5])

                        ground_truth.append({
                            'class_name': normalized_class,
                            'bbox': [x1, y1, x2, y2]
                        })

    return ground_truth


def get_frame_number_from_filename(filename):
    """
    Extract frame number from filename like 'frame_0001.jpg' or '0001.jpg'
    """
    # Remove extension and get the numeric part
    base_name = os.path.splitext(filename)[0]

    # Try different patterns
    if base_name.startswith('frame_'):
        return int(base_name[6:])  # for 'frame_0001'
    else:
        # Assume the filename is just the number
        return int(base_name)  # for '0001'


def create_detector(model_type, conf_threshold):
    """
    Create detector based on model type
    """
    model_config = config.MODELS_CONFIG.get(model_type)
    if not model_config:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == 'yolo':
        return YOLODetector(
            model_config['model'],
            model_config['config'],
            model_config['classes'],
            conf_threshold=conf_threshold
        )
    elif model_type == 'ssd':
        return SSDDetector(
            model_config['model'],
            model_config['config'],
            model_config['classes'],
            conf_threshold=conf_threshold
        )
    elif model_type == 'faster_rcnn':
        return FasterRCNNDetector(
            model_config['model'],
            model_config['config'],
            model_config['classes'],
            conf_threshold=conf_threshold
        )


def draw_detections(image, detections, vehicle_only=True):
    """
    Draw detections on image with different colors for each class
    """
    for detection in detections:
        class_name = detection['class_name']

        if vehicle_only and class_name not in config.VEHICLE_CLASSES:
            continue

        confidence = detection['confidence']
        bbox = detection['bbox']

        # Get color for this class
        color = config.VEHICLE_COLORS.get(class_name, (0, 255, 0))  # Default green

        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Draw label background
        label = f"{class_name}: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1] - 10),
                      (bbox[0] + label_size[0], bbox[1]), color, -1)

        # Draw label text
        cv2.putText(image, label, (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection using OpenCV DNN')
    parser.add_argument('--images_path', type=str, required=True, help='Path to images directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotation file (mov03478.txt)')
    parser.add_argument('--model', type=str, choices=['yolo', 'ssd', 'faster_rcnn',],
                        required=True, help='Model type for detection')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for detection')
    parser.add_argument('--show_display', action='store_true',
                        help='Show detection results in window (optional)')
    parser.add_argument('--save_output', action='store_true',
                        help='Save processed images with detections')
    parser.add_argument('--output_path', type=str, default='output',
                        help='Path to save processed images')
    parser.add_argument('--vehicle_only', action='store_true', default=True,
                        help='Detect only vehicle classes')

    args = parser.parse_args()

    # Create output directory if needed
    if args.save_output and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Create detector
    detector = create_detector(args.model, args.conf_threshold)
    # После создания детектора
    print(f"Model files check:")
    model_config = config.MODELS_CONFIG[args.model]
    print(f"Model path: {model_config['model']}, exists: {os.path.exists(model_config['model'])}")
    print(f"Config path: {model_config['config']}, exists: {os.path.exists(model_config['config'])}")
    print(f"Classes path: {model_config['classes']}, exists: {os.path.exists(model_config['classes'])}")

    # Get image files and sort them to maintain frame order
    image_files = glob.glob(os.path.join(args.images_path, "*.jpg")) + \
                  glob.glob(os.path.join(args.images_path, "*.png")) + \
                  glob.glob(os.path.join(args.images_path, "*.jpeg"))

    # Sort files by frame number
    image_files.sort(key=lambda x: get_frame_number_from_filename(os.path.basename(x)))

    if not image_files:
        print("No images found in the specified directory")
        return

    total_tpr = 0
    total_fdr = 0
    processed_images = 0

    print(f"Starting processing of {len(image_files)} frames...")
    print("Press 'q' to stop processing if display is enabled\n")

    for image_file in image_files:
        # Load image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Could not load image: {image_file}")
            continue

        # Get frame number from filename
        frame_number = get_frame_number_from_filename(os.path.basename(image_file))

        # Load ground truth for this specific frame
        ground_truth = load_ground_truth(args.annotation_file, frame_number,
                                         image.shape[1], image.shape[0])

        # Detect objects
        detections = detector.detect(image)

        print(f"=== DEBUG INFO for Frame {frame_number:04d} ===")
        print(f"DEBUG: Total detections before filtering: {len(detections)}")
        if detections:
            print(f"DEBUG: First few detections:")
            for i, det in enumerate(detections[:3]):  # первые 3 обнаружения
                print(f"  Detection {i}: Class: '{det['class_name']}', "
                      f"Confidence: {det['confidence']:.3f}, "
                      f"BBox: {det['bbox']}")
        else:
            print("DEBUG: No detections at all!")

        # Проверяем vehicle detections до фильтрации по confidence
        all_vehicle_detections = [det for det in detections
                                  if det['class_name'] in config.VEHICLE_CLASSES]
        print(f"DEBUG: Vehicle detections before confidence filter: {len(all_vehicle_detections)}")

        # Проверяем какие vehicle классы прошли confidence threshold
        confident_vehicle_detections = [det for det in all_vehicle_detections
                                        if det['confidence'] > args.conf_threshold]
        print(
            f"DEBUG: Vehicle detections after confidence filter ({args.conf_threshold}): {len(confident_vehicle_detections)}")
        print("=== END DEBUG ===")

        # Filter vehicle detections if requested
        if args.vehicle_only:
            vehicle_detections = [det for det in detections
                                  if det['class_name'] in config.VEHICLE_CLASSES]
        else:
            vehicle_detections = detections

        # Calculate metrics
        tpr, fdr = calculate_metrics(vehicle_detections, ground_truth)
        if ground_truth:
            gt_classes = [gt['class_name'] for gt in ground_truth]
            detected_classes = [det['class_name'] for det in vehicle_detections]

            print(f"DEBUG: Ground Truth classes: {set(gt_classes)}")
            print(f"DEBUG: Detected classes: {set(detected_classes)}")
            print(f"DEBUG: Ground Truth bboxes: {[gt['bbox'] for gt in ground_truth[:2]]}")  # первые 2
            if vehicle_detections:
                print(f"DEBUG: Detected bboxes: {[det['bbox'] for det in vehicle_detections[:2]]}")  # первые 2
        total_tpr += tpr
        total_fdr += fdr
        processed_images += 1

        # Progress indicator
        if processed_images % 100 == 0:
            print(f"Processed {processed_images}/{len(image_files)} frames...")

        # Only draw detections if we need to display or save
        if args.show_display or args.save_output:
            # Create a copy for drawing to avoid modifying original
            display_image = image.copy()
            draw_detections(display_image, detections, args.vehicle_only)

            # Add metrics and frame info to image
            metrics_text = f"Frame: {frame_number} | TPR: {tpr:.3f} | FDR: {fdr:.3f}"
            cv2.putText(display_image, metrics_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Save image if requested
            if args.save_output:
                output_filename = f"frame_{frame_number:04d}_detected.jpg"
                output_path = os.path.join(args.output_path, output_filename)
                cv2.imwrite(output_path, display_image)

            # Display image if requested
            if args.show_display:
                cv2.imshow('Vehicle Detection', display_image)
                # Wait for key press but don't block processing
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Processing stopped by user")
                    break

        # Print frame results
        print(f"Frame {frame_number:04d} | "
              f"Detections: {len(vehicle_detections):2d} | "
              f"Ground Truth: {len(ground_truth):2d} | "
              f"TPR: {tpr:.3f} | FDR: {fdr:.3f}")

    # Clean up display window
    if args.show_display:
        cv2.destroyAllWindows()

    # Calculate average metrics
    if processed_images > 0:
        avg_tpr = total_tpr / processed_images
        avg_fdr = total_fdr / processed_images

        print(f"\n=== Final Results ===")
        print(f"Processed frames: {processed_images}")
        print(f"Average TPR: {avg_tpr:.3f}")
        print(f"Average FDR: {avg_fdr:.3f}")

        # Calculate additional score for quality
        quality_score = 10 * avg_tpr
        print(f"Quality Score: {quality_score:.2f}/10")

        # Save summary to file
        summary_filename = f"results_{args.model}_conf{args.conf_threshold}.txt"
        summary_file = os.path.join(args.output_path if args.save_output else '.', summary_filename)
        with open(summary_file, 'w') as f:
            f.write("=== Vehicle Detection Results ===\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Confidence threshold: {args.conf_threshold}\n")
            f.write(f"Processed frames: {processed_images}\n")
            f.write(f"Average TPR: {avg_tpr:.3f}\n")
            f.write(f"Average FDR: {avg_fdr:.3f}\n")
            f.write(f"Quality Score: {quality_score:.2f}/10\n")

        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()