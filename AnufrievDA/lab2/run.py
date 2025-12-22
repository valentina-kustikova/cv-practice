import cv2
import os
import argparse
from detector_impl import SSDMobileNet, YOLOv4, FasterRCNN
from dataset_loader import AnnotationLoader
from performance_metrics import calculate_final_metrics

# Настройки моделей
MODELS_CONFIG = {
    'ssd': {
        'class': SSDMobileNet,
        'model': 'models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
        'config': 'models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
        'classes': 'models/coco_names.txt'
    },
    'yolo': {
        'class': YOLOv4,
        'model': 'models/yolo/yolov4-tiny.weights',
        'config': 'models/yolo/yolov4-tiny.cfg',
        'classes': 'models/coco_names.txt'
    },
    'rcnn': {
        'class': FasterRCNN,
        'model': 'models/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb',
        'config': 'models/faster_rcnn_resnet50_coco_2018_01_28/faster_rcnn_resnet50_coco_2018_01_28.pbtxt',
        'classes': 'models/coco_names.txt'
    }
}

def main():
    parser = argparse.ArgumentParser(description="Object Detection Lab 2")
    parser.add_argument('--model', type=str, required=True, choices=['ssd', 'yolo', 'rcnn'], help='Model type')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to images folder')
    parser.add_argument('--gt', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--show', action='store_true', help='Visualize results')
    args = parser.parse_args()

    # Инициализация
    if args.model not in MODELS_CONFIG:
        print(f"Error: Model {args.model} not found.")
        return

    cfg = MODELS_CONFIG[args.model]
    print(f"Initializing {args.model.upper()}...")
    
    detector = cfg['class'](cfg['model'], cfg['config'], cfg['classes'])

    # Загружаем разметку
    loader = AnnotationLoader(args.gt)
    
    frame_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Processing {len(frame_files)} frames...")

    dataset_gt = []
    dataset_preds = []

    for filename in frame_files:
        try:
            frame_id = int(os.path.splitext(filename)[0])
        except ValueError:
            continue

        img_path = os.path.join(args.img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        detections = detector.detect(image)
        
        # Получение эталонной разметки
        gt_boxes = loader.get_boxes(frame_id)

        dataset_gt.append(gt_boxes)
        dataset_preds.append(detections)

        if args.show:
            vis_img = detector.draw_results(image, detections)
            
            for g in gt_boxes:
                x1, y1, x2, y2 = g['box']
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(vis_img, g['class'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow(f"Detector: {args.model}", vis_img)
            if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
                break

    cv2.destroyAllWindows()

    print("\nCalculating metrics...")
    tpr, fdr, tp, fp, fn = calculate_final_metrics(dataset_gt, dataset_preds)

    print("=" * 40)
    print(f"RESULTS FOR MODEL: {args.model.upper()}")
    print("=" * 40)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 40)
    print(f"TPR (Recall): {tpr:.4f}")
    print(f"FDR:          {fdr:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()