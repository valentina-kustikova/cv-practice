import cv2
import os
import argparse
import json
from detectors.yolo_detector import YOLODetector
from detectors.ssd_detector import SSDMobileNetDetector
from utils.metrics import calculate_frame_metrics
import config
from utils.visualization import draw_detections as draw_detections_vis

VEHICLE_CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']

def load_ground_truth(annotation_path, img_width, img_height, vehicle_classes=None):
    ground_truths = {}

    if vehicle_classes is None:
        vehicle_classes = VEHICLE_CLASSES
    
    if not os.path.exists(annotation_path):
        print(f"Файл аннотации не найден: {annotation_path}")
        return ground_truths

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    print(f"Загружаем аннотации из: {annotation_path}")
    print(f"Всего строк в файле: {len(lines)}")

    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        values = line.split()
        if len(values) < 6:
            print(f"Строка {line_num}: неверное количество значений ({len(values)})")
            continue

        try:
            frame_id = int(values[0])
            class_name = values[1].strip().lower()
            if class_name not in vehicle_classes:
                continue

            x1 = int(float(values[2]))
            y1 = int(float(values[3]))
            x2 = int(float(values[4]))
            y2 = int(float(values[5]))
            
        except (ValueError, IndexError) as e:
            print(f"Строка {line_num}: ошибка парсинга - {e}")
            continue

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width - 1, x2)
        y2 = min(img_height - 1, y2)

        if x2 <= x1 or y2 <= y1:
            print(f"Строка {line_num}: неверные размеры bbox ({x1},{y1},{x2},{y2})")
            continue

        obj = {
            'class_name': class_name,
            'bbox': [x1, y1, x2, y2]
        }
        
        if frame_id not in ground_truths:
            ground_truths[frame_id] = []
        ground_truths[frame_id].append(obj)

    print(f"Загружено объектов по кадрам: {len(ground_truths)}")
    return ground_truths

def map_class_for_ssd(class_name):
    if class_name == 'motorcycle':
        return 'motorbike'  
    elif class_name == 'truck':
        return 'car'  
    elif class_name == 'bicycle':
        return 'bicycle'  
    else:
        return class_name  

def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection using OpenCV DNN')
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--model', type=str, choices=['yolo', 'ssd_mobilenet', 'yolo_tiny'], 
                       default='yolo', help='Model to use for detection')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')  
    parser.add_argument('--display', action='store_true', help='Display results')
    parser.add_argument('--frame_limit', type=int, default=-1, help='Limit number of frames to process')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for metrics')  
    
    args = parser.parse_args()
    
    try:
        model_config = config.MODELS[args.model]
    except (AttributeError, KeyError):
        print(f"Ошибка: модель '{args.model}' не найдена в config.MODELS")
        print("Доступные модели:", list(config.MODELS.keys()))
        return
    
    print(f"Используется модель: {args.model}")
    print(f"Порог уверенности: {args.confidence}")
    print(f"Порог IoU: {args.iou_threshold}")
    
    if 'yolo' in args.model:
        detector = YOLODetector(
            model_config['model'],
            model_config['config'],
            model_config['classes'],
            confidence_threshold=args.confidence
        )
    else:
        detector = SSDMobileNetDetector(
            model_config['model'],
            model_config['config'],
            model_config['classes'],
            confidence_threshold=args.confidence
        )
    
    image_files = sorted([f for f in os.listdir(args.images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if args.frame_limit > 0:
        image_files = image_files[:args.frame_limit]
    
    print(f"\nНайдено изображений: {len(image_files)}")
    print("-" * 70)
    
    first_image_path = os.path.join(args.images_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Ошибка: не удалось загрузить первое изображение {first_image_path}")
        return
    
    h, w = first_image.shape[:2]
    
    ground_truths = load_ground_truth(
        args.annotation_file,
        img_width=w,
        img_height=h,
        vehicle_classes=VEHICLE_CLASSES
    )
    
    frame_metrics_list = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(args.images_dir, image_file)
        
        print(f"\nОбработка изображения {i+1}/{len(image_files)}: {image_file} (кадр {i})")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка: не удалось загрузить изображение {image_path}")
            continue
        
        current_ground_truths = ground_truths.get(i, [])
        detections = detector.detect(image)
        
        if args.model == 'ssd_mobilenet':
            vehicle_detections = []
            for det in detections:
                class_name = det['class_name']
                if class_name in ['car', 'bus', 'motorbike', 'bicycle']:
                    det_copy = det.copy()
                    if det_copy['class_name'] == 'motorbike':
                        det_copy['class_name'] = 'motorcycle'
                    vehicle_detections.append(det_copy)
            
            vehicle_ground_truths = []
            for gt in current_ground_truths:
                gt_class = gt['class_name']
                if gt_class in ['car', 'bus', 'motorcycle', 'bicycle']:
                    vehicle_ground_truths.append(gt)
                elif gt_class == 'truck':
                    gt_copy = gt.copy()
                    gt_copy['class_name'] = 'car'
                    vehicle_ground_truths.append(gt_copy)
        else:
            vehicle_detections = [d for d in detections if d['class_name'] in VEHICLE_CLASSES]
            vehicle_ground_truths = [gt for gt in current_ground_truths if gt['class_name'] in VEHICLE_CLASSES]
        
        frame_metrics = calculate_frame_metrics(
            vehicle_detections, 
            vehicle_ground_truths, 
            args.iou_threshold
        )
        
        frame_metrics_list.append({
            'frame_id': i,
            'image_file': image_file,
            'detections_count': len(vehicle_detections),
            'ground_truth_count': len(vehicle_ground_truths),
            **frame_metrics
        })
        
        total_tp += frame_metrics['tp']
        total_fp += frame_metrics['fp']
        total_fn += frame_metrics['fn']
        
        print(f"Всего детекций: {len(detections)}")
        print(f"Транспортных средств в детекциях: {len(vehicle_detections)}")
        print(f"Транспортных средств в GT: {len(vehicle_ground_truths)}")
        print(f"TP: {frame_metrics['tp']}, FP: {frame_metrics['fp']}, FN: {frame_metrics['fn']}")
        print(f"TPR: {frame_metrics['tpr']:.4f}, FDR: {frame_metrics['fdr']:.4f}")
        
        if args.display:
            vis_detections = []
            for d in vehicle_detections:
                x1, y1, x2, y2 = d['bbox']
                try:
                    class_id = detector.classes.index(d['class_name'])
                except (ValueError, AttributeError):
                    class_id = 0
                vis_detections.append([
                    x1, y1, x2, y2,
                    class_id,
                    d['confidence'],
                    d['class_name']
                ])
            
            result_image = draw_detections_vis(image.copy(), vis_detections, VEHICLE_CLASSES)
            metrics_text = f"Frame: {i} | TPR: {frame_metrics['tpr']:.3f} | FDR: {frame_metrics['fdr']:.3f}"
            cv2.putText(result_image, metrics_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            file_text = f"File: {image_file}"
            cv2.putText(result_image, file_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            objects_text = f"Detected: {len(vehicle_detections)} | GT: {len(vehicle_ground_truths)}"
            cv2.putText(result_image, objects_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            for gt in vehicle_ground_truths:
                x1, y1, x2, y2 = gt['bbox']
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  
                class_text = gt['class_name']
                if gt['class_name'] == 'truck':
                    class_text = 'truck->car'  
                cv2.putText(result_image, f"GT:{class_text}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow('Vehicle Detection', result_image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("Прерывание по запросу пользователя")
                break
            elif key == ord(' '):
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord(' '):
                        break
                    elif key2 == ord('q'):
                        cv2.destroyAllWindows()
                        return
        
        print("-" * 50)
    
    # Вычисление итоговых метрик
    if frame_metrics_list:
        global_tpr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_fdr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        
        avg_tpr = sum(fm['tpr'] for fm in frame_metrics_list) / len(frame_metrics_list)
        avg_fdr = sum(fm['fdr'] for fm in frame_metrics_list) / len(frame_metrics_list)
        
        print(f"\n" + "="*70)
        print(f"=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
        print(f"Обработано изображений: {len(frame_metrics_list)}")
        print(f"Общее количество объектов: TP={total_tp}, FP={total_fp}, FN={total_fn}")
        print(f"Глобальный TPR (Recall): {global_tpr:.4f}")
        print(f"Глобальный FDR: {global_fdr:.4f}")
        print(f"Средний TPR по кадрам: {avg_tpr:.4f}")
        print(f"Средний FDR по кадрам: {avg_fdr:.4f}")
        
        print(f"\n=== ДЕТАЛЬНАЯ СТАТИСТИКА ПО КАДРАМ ===")
        for fm in frame_metrics_list[:10]:  
            print(f"Кадр {fm['frame_id']}: TPR={fm['tpr']:.4f}, FDR={fm['fdr']:.4f}, "
                  f"TP={fm['tp']}, FP={fm['fp']}, FN={fm['fn']}")
        if len(frame_metrics_list) > 10:
            print(f"... и еще {len(frame_metrics_list) - 10} кадров")
        
        results = {
            'model': args.model,
            'confidence_threshold': args.confidence,
            'iou_threshold': args.iou_threshold,
            'total_images': len(frame_metrics_list),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'global_tpr': float(global_tpr),
            'global_fdr': float(global_fdr),
            'avg_tpr': float(avg_tpr),
            'avg_fdr': float(avg_fdr),
            'frame_metrics': frame_metrics_list
        }
        
        output_file = f"results_{args.model}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nРезультаты сохранены в {output_file}")
        print("="*70)
    else:
        print("Не обработано ни одного изображения. Проверьте пути и форматы файлов.")
    
    if args.display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
