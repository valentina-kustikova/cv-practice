import os
import glob
import time
from typing import List, Tuple

import cv2 as cv
import numpy as np

from detector_base import Detection
from detector_factory import VehicleDetectorFactory
from annotation_loader import AnnotationLoader
from evaluator import DetectionEvaluator


class VehicleDetectionApp:
    """
    Демонстрационное приложение:
    - загрузка кадров;
    - запуск выбранной модели;
    - отображение детекций и Ground Truth;
    - вычисление TPR и FDR.
    """

    def __init__(self):
        self.gt_color = (255, 0, 255)  
        self.text_color = (255, 255, 255)
        self.class_colors = {}  

    def _get_color_for_class(self, class_name: str):
        if class_name not in self.class_colors:
            color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            self.class_colors[class_name] = color
        return self.class_colors[class_name]

    def _draw_detections(self, image, detections: List[Detection]):
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            class_name = det.class_name
            color = self._get_color_for_class(class_name)
            conf = det.confidence

            cv.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label_inside = f"{class_name}: {conf:.3f}"
            (tw, th), baseline = cv.getTextSize(
                label_inside, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x_inside = x1 + 2
            text_y_inside = y1 + th + 2

            cv.rectangle(
                result,
                (x1, y1),
                (x1 + tw + 4, y1 + th + baseline + 4),
                color,
                thickness=cv.FILLED
            )
            cv.putText(
                result,
                label_inside,
                (text_x_inside, text_y_inside),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

            # Наблюдаемый класс
            observed_label = class_name
            (tw2, th2), base2 = cv.getTextSize(
                observed_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x_above = x1
            text_y_above = max(0, y1 - 5)

            cv.putText(
                result,
                observed_label,
                (text_x_above, text_y_above),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        return result

    def _draw_ground_truth(self, image, gt_boxes: List[Tuple[int, int, int, int]]):
        result = image.copy()

        for bbox in gt_boxes:
            x1, y1, x2, y2 = bbox
            cv.rectangle(result, (x1, y1), (x2, y2), self.gt_color, 2)

            label = "GT"
            (tw, th), baseline = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x1
            text_y = max(0, y1 - 5)

            cv.rectangle(
                result,
                (text_x, text_y - th - baseline),
                (text_x + tw + 4, text_y + baseline),
                self.gt_color,
                thickness=cv.FILLED
            )
            cv.putText(
                result,
                label,
                (text_x + 2, text_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return result

    def _draw_metrics(self, image,
                      global_tpr: float,
                      global_fdr: float,
                      frame_tpr: float,
                      frame_fdr: float):
        result = image.copy()

        lines = [
            f"TPR: {global_tpr:.3f}",
            f"FDR: {global_fdr:.3f}",
            f"Frame TPR: {frame_tpr:.3f}",
            f"Frame FDR: {frame_fdr:.3f}",
        ]

        for i, text in enumerate(lines):
            cv.putText(
                result,
                text,
                (10, 30 + i * 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.text_color,
                2
            )

        return result

    def run(self,
            images_dir: str,
            annotations_path: str,
            model_key: str,
            confidence_threshold: float = 0.5,
            display: bool = True,
            show_ground_truth: bool = False):
        """
        Главный метод запуска:
        - images_dir — путь к кадрам;
        - annotations_path — путь к файлу разметки;
        - model_key — 'yolo' / 'ssd' / 'rcnn';
        - confidence_threshold — порог уверенности;
        - display — показывать окна с результатами;
        - show_ground_truth — рисовать GT-боксы.
        """

        available = VehicleDetectorFactory.get_available_models()
        if model_key not in available:
            print("Неизвестная модель.")
            print("Доступные модели:", ", ".join(available.keys()))
            return

        model_info = available[model_key]
        print(f"Выбрана модель: {model_info['name']}")
        print(f"Классы транспортных средств: {model_info['vehicle_classes']}")
        print(f"Отображение GT: {'ВКЛ' if show_ground_truth else 'ВЫКЛ'}")

        # Создаём детектор
        try:
            detector = VehicleDetectorFactory.create_detector(model_key, confidence_threshold)
        except Exception as e:
            print(f"Ошибка при создании детектора: {e}")
            print("Проверьте наличие файлов в папках models/ и configs/")
            return

        # Загрузка разметки + инициализация оценщика
        annotation_loader = AnnotationLoader(annotations_path, target_class="car")
        evaluator = DetectionEvaluator(target_class="car")

        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        print(f"Найдено изображений: {len(image_paths)}")

        total_time = 0.0
        processed_frames = 0

        for image_path in image_paths:
            image_id = os.path.splitext(os.path.basename(image_path))[0]

            image = cv.imread(image_path)
            if image is None:
                print(f"Не удалось прочитать изображение: {image_path}")
                continue

            gt_boxes = annotation_loader.get_ground_truth(image_id)

            start = time.time()
            detections = detector.detect(image)
            end = time.time()

            frame_time = end - start
            total_time += frame_time
            processed_frames += 1

            frame_tpr, frame_fdr = evaluator.evaluate_frame(detections, gt_boxes)
            global_tpr, global_fdr = evaluator.get_metrics()

            if display:
                frame_to_show = image.copy()

                if show_ground_truth:
                    frame_to_show = self._draw_ground_truth(frame_to_show, gt_boxes)

                frame_to_show = self._draw_detections(frame_to_show, detections)
                frame_to_show = self._draw_metrics(
                    frame_to_show,
                    global_tpr,
                    global_fdr,
                    frame_tpr,
                    frame_fdr
                )

                cv.imshow("Vehicle Detection", frame_to_show)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            print(
                f"{image_id}.jpg | time: {frame_time:.3f}s | "
                f"TPR: {global_tpr:.3f} | FDR: {global_fdr:.3f} | "
                f"frame TPR: {frame_tpr:.3f} | frame FDR: {frame_fdr:.3f}"
            )

        final_tpr, final_fdr = evaluator.get_metrics()
        avg_time = total_time / processed_frames if processed_frames > 0 else 0.0

        print("\n" + "=" * 50)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Модель: {model_info['name']}")
        print(f"Обработано кадров: {processed_frames}")
        print(f"Среднее время обработки: {avg_time:.3f}s")
        print(f"TPR: {final_tpr:.3f}")
        print(f"FDR: {final_fdr:.3f}")
        print(f"True Positives: {evaluator.true_positives}")
        print(f"False Positives: {evaluator.false_positives}")
        print(f"False Negatives: {evaluator.false_negatives}")

        if display:
            cv.destroyAllWindows()
