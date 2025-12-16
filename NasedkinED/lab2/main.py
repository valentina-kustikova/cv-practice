import argparse
import os

import cv2

from detectors import MobileNetSSDDetector, YOLOv8Detector
from utils import parse_ground_truth, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Детектирование транспорта")
    parser.add_argument("images_path", help="Путь до папки с кадрами")
    parser.add_argument("gt_path", help="Путь до файла разметки")
    parser.add_argument("--model", default="ssd", choices=["ssd", "yolov8"], help="Модель (ssd|yolov8)")
    parser.add_argument("--vis", action="store_true", help="Показывать видео")

    # Пути к весам (можно настроить под себя)
    parser.add_argument("--ssd_proto", default="models/MobileNetSSD_deploy.prototxt")
    parser.add_argument("--ssd_weights", default="models/MobileNetSSD_deploy.caffemodel")
    parser.add_argument("--yolo_cfg", default="models/yolov3.cfg")
    parser.add_argument("--yolo_weights", default="models/yolov3.weights")
    parser.add_argument("--yolo_names", default="models/coco.names")

    args = parser.parse_args()

    # Фабрика детекторов
    if args.model == "ssd":
        if not os.path.exists(args.ssd_proto):
            print("Ошибка: Не найден prototxt")
            return
        detector = MobileNetSSDDetector(args.ssd_weights, args.ssd_proto)
    else:
        detector = YOLOv8Detector("models/yolov8n.onnx", conf_threshold=0.4)

    ground_truth = parse_ground_truth(args.gt_path)
    if not ground_truth:
        print("Разметка не загружена или пуста.")
        return

    frame_ids = sorted(ground_truth.keys())
    total_tp, total_fp, total_fn = 0, 0, 0

    print(f"Начинаем обработку {len(frame_ids)} кадров...")

    for fid in frame_ids:
        # Подбираем имя файла (0.jpg, 1.jpg...)
        img_path = os.path.join(args.images_path, f"{fid:06d}.jpg")
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # 1. Детекция
        detections = detector.detect(frame)

        # 2. Метрики
        gt_boxes = ground_truth[fid]
        tpr, fdr, tp, fp, fn = compute_metrics(gt_boxes, detections)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 3. Визуализация
        if args.vis:
            # GT (Зеленый)
            for box in gt_boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "GT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Prediction (Синий)
            for det in detections:
                x, y, w, h = det['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.3f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Инфо панель
            cv2.rectangle(frame, (0, 0), (400, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Frame: {fid} | TPR: {tpr:.3f} | FDR: {fdr:.3f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Result", frame)
            if cv2.waitKey(1) == 27:  # ESC
                break
        else:
            # Логирование в консоль каждые 100 кадров, чтобы не спамить
            if fid % 100 == 0:
                print(f"Обработано кадров: {fid}/{len(frame_ids)}")

    # Итог
    if total_tp + total_fn > 0:
        global_tpr = total_tp / (total_tp + total_fn)
    else:
        global_tpr = 0

    if total_tp + total_fp > 0:
        global_fdr = total_fp / (total_tp + total_fp)
    else:
        global_fdr = 0

    print("=" * 40)
    print(f"ИТОГОВЫЕ МЕТРИКИ ({args.model.upper()}):")
    print(f"Global TPR: {global_tpr:.4f}")
    print(f"Global FDR: {global_fdr:.4f}")
    print("=" * 40)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
