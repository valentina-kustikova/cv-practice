import argparse
import os
import cv2
import numpy as np
from src.yolov3_detector import YOLOv3Detector
from src.utils import draw_predictions, load_ground_truth_labels, get_image_files
from src.metrics import calculate_tpr_fdr

COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
]

def main():
    parser = argparse.ArgumentParser(description="Object Detection with OpenCV DNN")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing image frames")
    parser.add_argument("--model_type", type=str, required=True, choices=["yolo"], help="Type of model to use (e.g., yolo)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file (.weights, .pb)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file (.cfg, .pbtxt)")
    parser.add_argument("--classes_path", type=str, required=True, help="Path to the classes names file (.names, .txt)")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the ground truth labels file (.txt)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--nms_threshold", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--display", action='store_true', help="Display the frames with detections")

    args = parser.parse_args()

    print(f"Initializing {args.model_type} detector...")
    if args.model_type.lower() == "yolo":
        detector = YOLOv3Detector(args.model_path, args.config_path, args.classes_path, args.conf_threshold, args.nms_threshold)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    classes = detector.classes

    print(f"Loading ground truth labels from {args.labels_path}...")
    ground_truths = load_ground_truth_labels(args.labels_path)

    print(f"Loading image files from {args.input_dir}...")
    image_files = get_image_files(args.input_dir)

    total_tpr = 0.0
    total_fdr = 0.0
    num_frames = len(image_files)

    print(f"Starting detection on {num_frames} frames...")

    for i, img_path in enumerate(image_files):
        print(f"Processing frame {i+1}/{num_frames}: {os.path.basename(img_path)}")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            continue

        base_filename = os.path.basename(img_path)
        boxes, class_ids, confidences = detector.detect(image)
        detections_formatted = [{'bbox': box, 'class': class_ids[j], 'confidence': confidences[j]} for j, box in enumerate(boxes)]

        gt_formatted = ground_truths.get(base_filename, [])

        frame_tpr, frame_fdr = calculate_tpr_fdr(detections_formatted, gt_formatted)
        # print(f"  Frame TPR: {frame_tpr:.3f}, FDR: {frame_fdr:.3f}")

        total_tpr += frame_tpr
        total_fdr += frame_fdr

        if args.display:
            output_image = draw_predictions(image, boxes, class_ids, confidences, classes, COLORS)
            metrics_text = f"Frame TPR: {frame_tpr:.3f}, FDR: {frame_fdr:.3f}"
            cv2.putText(output_image, metrics_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.imshow('Object Detection', output_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break # Выход

    if num_frames > 0:
        avg_tpr = total_tpr / num_frames
        avg_fdr = total_fdr / num_frames
        print(f"\n--- Results ---")
        print(f"Average TPR: {avg_tpr:.3f}")
        print(f"Average FDR: {avg_fdr:.3f}")
    else:
        print("\nNo frames were processed.")

    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
