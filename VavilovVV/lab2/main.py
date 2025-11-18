import argparse
import cv2
import glob
import os
import time

from detectors import VehicleDetectorFactory
from utils import (
    AnnotationLoader,
    DetectionEvaluator,
    draw_detections,
    draw_ground_truth,
    display_metrics,
)

class VehicleDetectionApp:
    def __init__(self):
        self.detection_color = (0, 255, 0)
        self.ground_truth_color = (255, 0, 255)

    def run(
        self,
        images_path: str,
        annotation_path: str,
        model_type: str,
        confidence_threshold: float = 0.5,
        display: bool = True,
        show_ground_truth: bool = False
    ):
        available_models = VehicleDetectorFactory.get_available_models()
        if model_type not in available_models:
            print(f"Доступные модели: {list(available_models.keys())}")
            return

        print(f"Выбрана модель: {available_models[model_type]['name']}")
        print(f"Классы транспортных средств: {available_models[model_type]['vehicle_classes']}")
        print(f"Отображение Ground Truth: {'ВКЛ' if show_ground_truth else 'ВЫКЛ'}")

        try:
            detector = VehicleDetectorFactory.create_detector(model_type, confidence_threshold)
        except Exception as e:
            print(f"Ошибка при создании детектора: {e}")
            print("Убедитесь, что модели загружены в папку 'models/'")
            return

        annotation_loader = AnnotationLoader(annotation_path)
        evaluator = DetectionEvaluator(target_class="car")

        image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
        print(f"Найдено изображений: {len(image_files)}")

        total_time = 0
        processed_frames = 0

        for image_file in image_files:
            image_id = os.path.splitext(os.path.basename(image_file))[0]
            image = cv2.imread(image_file)
            if image is None:
                print(f"Не удалось загрузить изображение: {image_file}")
                continue

            ground_truth = annotation_loader.get_ground_truth(image_id)

            start_time = time.time()
            detections = detector.detect(image)
            end_time = time.time()

            processing_time = end_time - start_time
            total_time += processing_time
            processed_frames += 1

            frame_tpr, frame_fdr = evaluator.evaluate_frame(detections, ground_truth)

            overall_tpr, overall_fdr = evaluator.get_metrics()

            if display:
                result_image = image.copy()
                if show_ground_truth:
                    result_image = draw_ground_truth(result_image, ground_truth, self.ground_truth_color)
                result_image = draw_detections(result_image, detections)

                result_image = display_metrics(result_image, overall_tpr, overall_fdr, frame_tpr, frame_fdr)

                cv2.imshow("Vehicle Detection", result_image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break

            print(
                f"Обработано: {image_id}.jpg | Время: {processing_time:.3f}s | "
                f"TPR: {overall_tpr:.3f} | FDR: {overall_fdr:.3f} | "
                f"Frame TPR: {frame_tpr:.3f} | Frame FDR: {frame_fdr:.3f}"
            )

        final_tpr, final_fdr = evaluator.get_metrics()
        avg_time = total_time / processed_frames if processed_frames > 0 else 0

        print("\n" + "=" * 50)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Модель: {available_models[model_type]['name']}")
        print(f"Обработано кадров: {processed_frames}")
        print(f"Среднее время обработки: {avg_time:.3f}s")
        print(f"TPR (True Positive Rate): {final_tpr:.3f}")
        print(f"FDR (False Discovery Rate): {final_fdr:.3f}")
        print(f"True Positives: {evaluator.true_positives}")
        print(f"False Positives: {evaluator.false_positives}")
        print(f"False Negatives: {evaluator.false_negatives}")

        if display:
            cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection using OpenCV DNN")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations file")
    parser.add_argument("--model", type=str, required=True, choices=["yolov4","yolov4tiny","mobilenet", "rcnn"], help="Model type for detection")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--no-display", action="store_true", help="Disable image display")
    parser.add_argument("--show-ground-truth", action="store_true", help="Display ground truth bounding boxes")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = VehicleDetectionApp()
    app.run(
        images_path=args.images,
        annotation_path=args.annotations,
        model_type=args.model,
        confidence_threshold=args.confidence,
        display=not args.no_display,
        show_ground_truth=args.show_ground_truth
    )