import cv2
import os
import glob
import time
from detector import VehicleDetector
from utils import load_ground_truth, draw_detections


class VideoPlayer:
    def __init__(self, images_path, annotations_path, detector, delay=30):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.detector = detector
        self.delay = delay  # Задержка между кадрами в мс
        self.paused = False
        self.current_frame = 0

        # Получаем список изображений
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(images_path, ext)))

        self.image_files.sort()  # Сортируем по имени
        self.total_frames = len(self.image_files)

        print(f"Loaded {self.total_frames} frames")

    def run(self):
        if self.total_frames == 0:
            print("No images found!")
            return

        print("\nControls:")
        print("  SPACE - Play/Pause")
        print("  → - Next frame")
        print("  ← - Previous frame")
        print("  S - Save current frame")
        print("  Q - Quit")
        print("  R - Reset to first frame")
        print("  + - Increase speed")
        print("  - - Decrease speed")
        print("  D - Toggle display info")

        self.display_info = True
        self.speed = 1.0  # Множитель скорости

        while self.current_frame < self.total_frames:
            if not self.paused:
                # Автоматическое воспроизведение
                self.process_frame()
                self.current_frame += 1

                # Задержка с учетом скорости
                actual_delay = max(1, int(self.delay / self.speed))
                key = cv2.waitKey(actual_delay) & 0xFF
            else:
                # В режиме паузы ждем дольше
                key = cv2.waitKey(100) & 0xFF

            # Обработка клавиш
            if self.handle_keypress(key):
                break

            # Обновление заголовка окна
            self.update_window_title()

    def process_frame(self):
        if self.current_frame >= self.total_frames:
            return

        image_file = self.image_files[self.current_frame]
        image = cv2.imread(image_file)

        if image is None:
            print(f"Error loading frame {self.current_frame}: {image_file}")
            return

        # Детектирование
        detections = self.detector.detect_vehicles(image)

        # Фильтрация транспортных средств
        vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
        vehicle_detections = [d for d in detections if d['class_name'] in vehicle_classes]

        # Загрузка ground truth
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        annotation_file = os.path.join(self.annotations_path, f"{image_name}.txt")
        ground_truth = load_ground_truth(annotation_file, image.shape[1], image.shape[0])

        # Вычисление метрик
        if ground_truth:
            tpr, fdr = self.detector.calculate_metrics(vehicle_detections, ground_truth)
        else:
            tpr, fdr = 0, 0

        # Отрисовка
        result_image = draw_detections(image.copy(), vehicle_detections, ground_truth)

        # Добавление информации
        if self.display_info:
            info_lines = [
                f"Frame: {self.current_frame + 1}/{self.total_frames}",
                f"Detections: {len(vehicle_detections)}",
                f"Speed: {self.speed:.1f}x"
            ]

            if ground_truth:
                info_lines.extend([
                    f"TPR: {tpr:.3f}",
                    f"FDR: {fdr:.3f}",
                    f"Ground truth: {len(ground_truth)}"
                ])

            for i, text in enumerate(info_lines):
                y_position = 30 + i * 25
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(result_image, text, (10, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Легенда
            cv2.putText(result_image, "Green: Detections",
                        (10, result_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result_image, "Blue: Ground Truth",
                        (10, result_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Изменение размера если нужно
        height, width = result_image.shape[:2]
        if width > 1200 or height > 800:
            scale = min(1200 / width, 800 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            result_image = cv2.resize(result_image, (new_width, new_height))

        cv2.imshow('Vehicle Detection', result_image)

        # Вывод информации в консоль каждые 10 кадров
        if self.current_frame % 10 == 0:
            print(f"Frame {self.current_frame + 1}/{self.total_frames} - "
                  f"Detections: {len(vehicle_detections)}, "
                  f"TPR: {tpr:.3f}, FDR: {fdr:.3f}")

    def handle_keypress(self, key):
        if key == ord('q') or key == 27:  # Q или ESC
            return True
        elif key == ord(' '):  # SPACE - пауза/продолжить
            self.paused = not self.paused
            print("Paused" if self.paused else "Playing")
        elif key == 83 or key == 3:  # Стрелка вправо
            self.paused = True
            self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            self.process_frame()
        elif key == 81 or key == 2:  # Стрелка влево
            self.paused = True
            self.current_frame = max(self.current_frame - 1, 0)
            self.process_frame()
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
        elif key == ord('d'):  # D - переключить информацию
            self.display_info = not self.display_info
            print(f"Display info: {'ON' if self.display_info else 'OFF'}")

        return False

    def save_current_frame(self):
        if self.current_frame < self.total_frames:
            image_file = self.image_files[self.current_frame]
            output_path = os.path.splitext(image_file)[0] + '_detection.jpg'

            # Перерисовываем текущий кадр и сохраняем
            image = cv2.imread(image_file)
            detections = self.detector.detect_vehicles(image)
            vehicle_detections = [d for d in detections if
                                  d['class_name'] in ['car', 'bus', 'truck', 'motorcycle', 'bicycle']]

            image_name = os.path.splitext(os.path.basename(image_file))[0]
            annotation_file = os.path.join(self.annotations_path, f"{image_name}.txt")
            ground_truth = load_ground_truth(annotation_file, image.shape[1], image.shape[0])

            result_image = draw_detections(image.copy(), vehicle_detections, ground_truth)
            cv2.imwrite(output_path, result_image)
            print(f"Frame saved to: {output_path}")

    def update_window_title(self):
        status = "PAUSED" if self.paused else "PLAYING"
        title = f"Vehicle Detection - {status} - Frame {self.current_frame + 1}/{self.total_frames} - {self.speed:.1f}x"
        cv2.setWindowTitle('Vehicle Detection', title)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Video Player for Vehicle Detection')
    parser.add_argument('--images_path', type=str, default='./dataset/images', help='Path to images directory')
    parser.add_argument('--annotations_path', type=str, default='./dataset/annotations',
                        help='Path to annotations directory')
    parser.add_argument('--model_type', type=str, default='yolo', choices=['yolo', 'ssd'])
    parser.add_argument('--model_path', type=str, default='models_data/yolov4-tiny.weights')
    parser.add_argument('--config_path', type=str, default='models_data/yolov4-tiny.cfg')
    parser.add_argument('--classes_path', type=str, default='models_data/coco.names')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--delay', type=int, default=30, help='Delay between frames in ms')

    args = parser.parse_args()

    # Создаем детектор
    detector = VehicleDetector(
        model_type=args.model_type,
        model_path=args.model_path,
        config_path=args.config_path,
        classes_file=args.classes_path
    )
    detector.set_confidence_threshold(args.confidence)

    # Запускаем видеоплеер
    player = VideoPlayer(
        images_path=args.images_path,
        annotations_path=args.annotations_path,
        detector=detector,
        delay=args.delay
    )

    try:
        player.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()