import cv2
import os
import argparse
from detectors import SSDDetector, YOLODetector
from evaluate import evaluate_frame

def parse_ground_truth(txt_path):
    gt = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            frame_id = int(parts[0])
            x1, y1, x2, y2 = map(int, parts[2:6])
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append({
                'box': [x1, y1, x2 - x1, y2 - y1],
                'class_name': 'CAR'
            })
    return gt

def draw_boxes(image, results, colors=None, is_gt=False):
    for res in results:
        x, y, w, h = res['box']
        if is_gt:
            color = (0, 255, 0)
            label = "CAR"
            cv2.putText(image, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            color = tuple(map(int, colors[res['class_id']]))
            label = f"{res['class_name']}: {res['confidence']:.3f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Путь к папке с кадрами и txt')
    parser.add_argument('--model', choices=['ssd', 'yolo'], required=True)
    parser.add_argument('--display', action='store_true', help='Показывать видео')
    args = parser.parse_args()

    gt = parse_ground_truth(os.path.join(args.data_path, 'mov03478.txt'))

    if args.model == 'ssd':
        detector = SSDDetector(
            model_path='models/ssd/MobileNetSSD_deploy.caffemodel',
            config_path='models/ssd/MobileNetSSD_deploy.prototxt.txt'
        )
    elif args.model == 'yolo':
        detector = YOLODetector(
            model_path='models/yolo/yolov3.weights',
            config_path='models/yolo/yolov3.cfg',
            names_path='models/yolo/coco.names'
        )

    total_tp = total_fp = total_fn = 0
    images = sorted([f for f in os.listdir(args.data_path) if f.lower().endswith('.jpg')])

    for img_name in images:
        frame_id = int(os.path.splitext(img_name)[0])
        image = cv2.imread(os.path.join(args.data_path, img_name))
        if image is None:
            continue

        predictions = detector.detect(image, conf_threshold=0.5, nms_threshold=0.4)
        ground_truth = gt.get(frame_id, [])

        tp, fp, fn = evaluate_frame(ground_truth, predictions)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if args.display:
            draw_boxes(image, ground_truth, is_gt=True)
            draw_boxes(image, predictions, detector.colors, is_gt=False)
            cv2.imshow(f'{args.model.upper()} Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

    tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fdr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    print("\n" + "="*50)
    print(f"Результаты модели {args.model.upper()}")
    print("="*50)
    print(f"TPR (Recall) : {tpr:.3f}")
    print(f"FDR          : {fdr:.3f}")
    print(f"Кадров       : {len(images)}")
    print("="*50)

if __name__ == "__main__":
    main()