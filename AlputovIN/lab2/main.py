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
    
    images = load_images(args.images)
    if not images:
        print(f"Ошибка: не найдено изображений в папке {args.images}", file=sys.stderr)
        sys.exit(1)
    
    annotations = load_annotations(args.annotations)

    all_detections = []
    all_gts = []

    total = len(images)
    print(f"Обработка {total} изображений...")
    for idx, (img_name, img) in enumerate(images.items(), 1):
        if idx % 100 == 0 or idx == 1:
            print(f"Обработано: {idx}/{total} ({idx*100//total}%)")
        detections = detector.detect(img)
        gt = annotations.get(img_name, [])
        all_detections.append(detections)
        all_gts.append(gt)
        img_vis = draw_boxes(img.copy(), detections)
        if args.show:
            cv2.imshow('Detection', img_vis)
            cv2.waitKey(1)
    if args.show:
        cv2.destroyAllWindows()

    tpr, fdr = calculate_metrics(all_detections, all_gts)
    print(f'TPR={tpr:.3f}, FDR={fdr:.3f}')

if __name__ == '__main__':
    main()
