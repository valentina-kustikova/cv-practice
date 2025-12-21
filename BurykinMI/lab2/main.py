import cv2
import os
import argparse
import glob
from detectors import SSDDetector, YOLODetector, YOLOv4TinyDetector
from utils import Metrics, parse_ground_truth


# ============================================================================
# Точка входа в приложение (Main Application):
# - Обработка аргументов командной строки
# - Инициализация выбранной модели (SSD или YOLO)
# - Основной цикл обработки видеопоследовательности
# - Визуализация результатов и вывод метрик в реальном времени
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Детектирование объектов (Транспорт)")
    parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "yolo", "yolov4-tiny"])
    parser.add_argument("--frames_dir", type=str, default="data/MOV03478")
    parser.add_argument("--gt_file", type=str, default="data/MOV03478/mov03478.txt")

    # Пути к моделям (укажите свои актуальные пути, если отличаются)
    parser.add_argument("--ssd_config", default="models/ssd/MobileNetSSD_deploy.prototxt.txt")
    parser.add_argument("--ssd_weights", default="models/ssd/MobileNetSSD_deploy.caffemodel")

    parser.add_argument("--yolo_config", default="models/yolo/yolov3.cfg")
    parser.add_argument("--yolo_weights", default="models/yolo/yolov3.weights")
    parser.add_argument("--yolo_classes", default="models/yolo/coco.names")

    parser.add_argument("--tiny_config", default="models/yolo/yolov4-tiny.cfg")
    parser.add_argument("--tiny_weights", default="models/yolo/yolov4-tiny.weights")

    args = parser.parse_args()

    # 1. Инициализация детектора (Factory Logic)
    print(f"Инициализация модели: {args.model.upper()}...")
    if args.model == "ssd":
        if not os.path.exists(args.ssd_weights):
            print("Ошибка: Веса SSD не найдены!")
            return
        detector = SSDDetector(args.ssd_config, args.ssd_weights)

    elif args.model == "yolo":
        if not os.path.exists(args.yolo_weights):
            print(f"Ошибка: Веса YOLO не найдены: {args.yolo_weights}")
            return
        detector = YOLODetector(args.yolo_config, args.yolo_weights, args.yolo_classes)

    elif args.model == "yolov4-tiny":
        if not os.path.exists(args.tiny_weights):
            print(f"Ошибка: Веса YOLOv4-Tiny не найдены: {args.tiny_weights}")
            return
        # Используем те же классы COCO, что и для обычного YOLO
        detector = YOLOv4TinyDetector(args.tiny_config, args.tiny_weights, args.yolo_classes)

    # 2. Загрузка разметки
    gt_data = parse_ground_truth(args.gt_file)
    metrics_calc = Metrics()

    # 3. Поиск изображений
    image_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    print(f"Найдено кадров: {len(image_paths)}")

    if not image_paths:
        print("Изображения не найдены. Проверьте путь.")
        return

    window_name = "Object Detection Lab"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # 4. Главный цикл
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None: continue

        # Извлекаем номер кадра (000005.jpg -> 5)
        try:
            frame_num = int(os.path.splitext(os.path.basename(img_path))[0])
        except ValueError:
            continue

        # Детектирование
        predictions = detector.detect(frame)

        # Получение эталона (GT)
        current_gt = gt_data.get(frame_num, [])

        # Обновление метрик
        metrics_calc.update(predictions, current_gt)
        curr_tpr, curr_fdr = metrics_calc.get_metrics()

        # Отрисовка
        # Ground Truth (Зеленый)
        for gt in current_gt:
            lbl, gx, gy, gw, gh = gt
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 1)
            cv2.putText(frame, f"GT: {lbl}", (gx, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Predictions (Цветной)
        for pred in predictions:
            lbl, conf, x, y, w, h = pred
            color = detector.get_color(lbl)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Умное позиционирование текста
            text_y = y + 20 if y + 20 < y + h else y + h - 5
            cv2.putText(frame, f"{lbl}: {conf:.2f}", (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Инфо-панель
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        stats = f"Frame: {frame_num} | {args.model.upper()} | TPR: {curr_tpr:.3f} | FDR: {curr_fdr:.3f}"
        cv2.putText(frame, stats, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Object Detection Lab", frame)

        # Клавиши: ESC - выход, Пробел - пауза
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    final_tpr, final_fdr = metrics_calc.get_metrics()
    print(f"\n--- ИТОГОВЫЕ РЕЗУЛЬТАТЫ ---\nМодель: {args.model.upper()}\nTPR: {final_tpr:.4f}\nFDR: {final_fdr:.4f}")


if __name__ == "__main__":
    main()
