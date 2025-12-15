import argparse
from detectors.detector_factory import get_detector
from utils import load_images, load_annotations, draw_boxes, calculate_metrics
import os
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(description='Object Detection Lab2')
    parser.add_argument('--images', type=str, default='data/imgs_MOV03478', help='Path to images folder')
    parser.add_argument('--annotations', type=str, default='data/mov03478.txt', help='Path to annotation file')
    parser.add_argument('--model', type=str, required=True, choices=['yolov8', 'fasterrcnn', 'nanodet'], help='Model name')
    parser.add_argument('--show', action='store_true', help='Show images with detections')
    args = parser.parse_args()

    # Проверка существования путей
    if not os.path.exists(args.images):
        print(f"Ошибка: папка с изображениями не найдена: {args.images}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.annotations):
        print(f"Ошибка: файл с аннотациями не найден: {args.annotations}", file=sys.stderr)
        sys.exit(1)

    try:
        detector = get_detector(args.model)
    except Exception as e:
        print(f"Ошибка при загрузке детектора: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Загружаем изображения
    images = load_images(args.images)
    if not images:
        print(f"Ошибка: не найдено изображений в папке {args.images}", file=sys.stderr)
        sys.exit(1)

    # 2. Загружаем аннотации, передавая путь к изображениям для корректного сопоставления имен
    # ВАЖНО: Это требует обновленного utils.py
    annotations = load_annotations(args.annotations, args.images)

    all_detections = []
    all_gts = []

    total = len(images)
    print(f"Обработка {total} изображений с использованием {args.model}...")

    # Используем sorted keys для детерминированного порядка
    for idx, img_name in enumerate(sorted(images.keys()), 1):
        img = images[img_name]

        if idx % 100 == 0 or idx == 1:
            print(f"Обработано: {idx}/{total} ({idx*100//total}%)")

        detections = detector.detect(img)

        # Получаем GT для текущего файла (если нет - пустой список)
        gt = annotations.get(img_name, [])

        all_detections.append(detections)
        all_gts.append(gt)

        if args.show:
            img_vis = draw_boxes(img.copy(), detections)

            # Опционально: Рисуем GT желтым пунктиром для отладки
            for g in gt:
                gx1, gy1, gx2, gy2 = g['bbox']
                cv2.rectangle(img_vis, (gx1, gy1), (gx2, gy2), (0, 255, 255), 1)

            cv2.imshow('Detection', img_vis)
            # Если нажата ESC (27) или Q (113) - выход
            key = cv2.waitKey(1)
            if key in [27, 113, ord('q')]:
                break

    if args.show:
        cv2.destroyAllWindows()

    tpr, fdr = calculate_metrics(all_detections, all_gts)
    print(f'TPR={tpr:.3f}, FDR={fdr:.3f}')

if __name__ == '__main__':
    main()