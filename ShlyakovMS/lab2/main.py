import os
import cv2
import numpy as np
import argparse
import sys
from detector import YOLOv3Detector, YOLOv4TinyDetector, MobileNetSSDDetector

def parse_labels(label_file):
    labels = {}
    if not os.path.exists(label_file):
        print(f"Файл разметки не найден: {label_file}")
        return labels
    with open(label_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            frame_id = int(parts[0])
            class_name = parts[1].lower()
            x1, y1, x2, y2 = map(int, parts[2:])
            box = [x1, y1, x2 - x1, y2 - y1]
            if frame_id not in labels:
                labels[frame_id] = []
            labels[frame_id].append({'class_name': class_name, 'box': box})
    return labels

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2); yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2); yi2 = min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0

def evaluate_detections(detections, gt_boxes, iou_thr=0.5):
    if not gt_boxes:
        return (1.0 if not detections else 0.0), (0.0 if not detections else 1.0)
    tp = fp = 0
    matched = [False] * len(gt_boxes)
    for det in detections:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if matched[i] or gt['class_name'] != det['class_name']: continue
            curr_iou = iou(det['box'], gt['box'])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_idx = i
        if best_iou >= iou_thr:
            tp += 1
            matched[best_idx] = True
        else:
            fp += 1
    fn = len(gt_boxes) - sum(matched)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    return tpr, fdr

def draw_detections(image, detections, class_to_color):
    for det in detections:
        x, y, w, h = det['box']
        class_name = det['class_name']
        conf = det['confidence']
        color = class_to_color.get(class_name, (0, 255, 0))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        cv2.putText(image, f"{class_name} {conf:.3f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, class_name.upper(), (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image

def create_video(frames_with_text, output_path, fps=30.0):
    if not frames_with_text:
        print(f"[SKIP] Нет кадров для видео: {output_path}")
        return
    h, w = frames_with_text[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames_with_text:
        out.write(frame)
    out.release()
    print(f"Видео сохранено: {output_path} ({len(frames_with_text)} кадров, {fps} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Практическая №2 – Детектирование ТС + видео из всех кадров")
    parser.add_argument('--image_dir', type=str, help='Папка с кадрами')
    parser.add_argument('--label_file', type=str, help='Файл с разметкой')
    parser.add_argument('--model', type=str, choices=['yolov3','yolov4tiny','ssd'], help='Модель')
    parser.add_argument('--output_dir', type=str, default='results', help='Папка для результатов')

    if len(sys.argv) == 1 or '--wdir' in sys.argv or 'spyder' in sys.executable.lower():
        print("Запуск в IDE — используем встроенные параметры")
        args = parser.parse_args([
            '--image_dir', r'C:\Users\maxsl\lab2\imgs_MOV03478',
            '--label_file', r'C:\Users\maxsl\lab2\mov03478.txt',
            '--model', 'yolov4tiny',
            '--output_dir', 'results'
        ])
    else:
        args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    best_dir = os.path.join(args.output_dir, 'best')
    worst_dir = os.path.join(args.output_dir, 'worst')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    print(f"Модель: {args.model.upper()}")
    print(f"Кадры: {args.image_dir}\n")

    if args.model == "yolov3":
        detector = YOLOv3Detector()
    elif args.model == "yolov4tiny":
        detector = YOLOv4TinyDetector()
    else:
        detector = MobileNetSSDDetector()

    class_to_color = {cls: tuple(map(int, col)) for cls, col in zip(detector.classes, detector.colors)}
    all_labels = parse_labels(args.label_file)

    images = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png'))])
    results = []
    video_frames = []

    print("Обработка кадров...")
    for i, img_file in enumerate(images, 1):
        frame_id = int(os.path.splitext(img_file)[0])
        img_path = os.path.join(args.image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        detections = detector.detect(img, conf_threshold=0.5, nms_threshold=0.4)
        for d in detections:
            d['color'] = class_to_color.get(d['class_name'], (0, 255, 0))

        gt = all_labels.get(frame_id, [])
        tpr, fdr = evaluate_detections(detections, gt)

        frame_out = draw_detections(img.copy(), detections, class_to_color)
        txt = f"Frame {frame_id:06d} | TPR={tpr:.4f} | FDR={fdr:.4f} | Detections={len(detections)}"
        cv2.putText(frame_out, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        video_frames.append(frame_out)

        results.append({
            'frame_id': frame_id,
            'filename': img_file,
            'image': img.copy(),
            'detections': detections,
            'tpr': tpr,
            'fdr': fdr
        })

        print(f"  [{i}/{len(images)}] {img_file}  TPR={tpr:.4f}; FDR={fdr:.4f}")

    results.sort(key=lambda x: x['tpr'], reverse=True)
    best = results[:5]
    worst = results[-5:]

    for rank, item in enumerate(best, 1):
        out = draw_detections(item['image'], item['detections'], class_to_color)
        cv2.putText(out, f"BEST #{rank} | TPR={item['tpr']:.4f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)
        cv2.imwrite(os.path.join(best_dir, f"best_{rank:02d}_{item['filename']}"), out)

    for rank, item in enumerate(worst, 1):
        out = draw_detections(item['image'], item['detections'], class_to_color)
        cv2.putText(out, f"WORST #{rank} | TPR={item['tpr']:.4f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 4)
        cv2.imwrite(os.path.join(worst_dir, f"worst_{rank:02d}_{item['filename']}"), out)

    video_path = os.path.join(args.output_dir, f"detection_result_{args.model}.mp4")
    create_video(video_frames, video_path, fps=30.0)

    avg_tpr = np.mean([r['tpr'] for r in results])
    avg_fpr = np.mean([r['fpr'] for r in results])

    print("\n" + "="*70)
    print(f"ГОТОВО! Обработано кадров: {len(results)}")
    print(f"Средний TPR  = {avg_tpr:.4f} → +{10*avg_tpr:.1f} баллов!")
    print(f"Средний FPR  = {avg_fpr:.4f}")
    print(f"Видео из всех кадров: {video_path}")
    print(f"Лучшие кадры {best_dir}")
    print(f"Худшие кадры {worst_dir}")
    print("="*70)

    results_dir = r"C:\Users\maxsl\lab2\results"
    os.makedirs(results_dir, exist_ok=True)
    
    report_path = os.path.join(results_dir, f"{args.model}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write(f"Модель: {args.model}\n")
        f.write(f"Средний TPR  = {avg_tpr:.4f} \n")
        f.write(f"Средний FPR  = {avg_fpr:.4f}\n")
 
    
    print(f"Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()