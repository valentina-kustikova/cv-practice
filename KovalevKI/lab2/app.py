import argparse
import cv2
import numpy as np
from pathlib import Path
from detectors import DETECTORS
from utils.io import load_annotations_custom
from utils.metrics import FrameLevelDetectionEvaluator
import time

def draw_detection(frame, box, label, conf, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"{conf:.3f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    t0 = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/images", help="Папка с кадрами (00000.jpg и т.д.)")
    parser.add_argument("--annotations", default="data/labels.txt", help="Файл labels.txt")
    parser.add_argument("--detector", choices=["mobilenet_ssd", "yolov5s_onnx"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", help="Только для SSD и YOLOv4")
    parser.add_argument("--classes", help="Файл классов (опционально)")
    parser.add_argument("--show", type=int, default=1, help="Визуализировать?")
    parser.add_argument("--delay", type=int, default=1)
    args = parser.parse_args()

    Detector = DETECTORS[args.detector]
    detector = Detector(
        model_path=args.model,
        config_path=args.config,
        classes_path=args.classes,
        conf_threshold=0.4,
        nms_threshold=0.4
    )

    ann_by_id = load_annotations_custom(args.annotations)

    img_paths = sorted(Path(args.images).glob("*.jpg"))
    id_to_path = {p.stem: p for p in img_paths}

    all_ids = sorted(
        set(ann_by_id.keys()) | set(id_to_path.keys()),
        key=lambda x: int(x) if x.isdigit() else x
    )
    
    tpr_list, fdr_list = [], []

    evaluator = FrameLevelDetectionEvaluator(target_class="car", iou_threshold=0.5)

    for fid in all_ids:
        gt_boxes, gt_labels = [], []
        for ann in ann_by_id.get(fid, []):
            label, x1, y1, w, h = ann
            if isinstance(label, str) and label.upper() == "CAR":
                gt_boxes.append((x1, y1, w, h))
                gt_labels.append("car")

        boxes, confs, cids = detector.detect(frame)
        det_boxes, det_labels, det_confs = [], [], []
        for box, conf, cid in zip(boxes, confs, cids):
            name = detector.get_class_name(cid).lower()
            color = (255, 0, 0)
            if name == "car" or cid == 2:
                det_boxes.append(box)
                det_labels.append(name)
                det_confs.append(conf)
                color = (0, 255, 0)
            if args.show:
                draw_detection(frame, box, name, conf, color)

        evaluator.accumulate_frame(det_boxes, det_labels, det_confs, gt_boxes, gt_labels)

        print(f"{fid}") # → TPR: {tpr_frame:.3f}, FDR: {fdr_frame:.3f}

        if args.show:
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    tpr, fdr = evaluator.get_metrics()
    elapsed = time.time() - t0
    print(f"Время выполнения: {elapsed:.2f} с ")
    print(f"\n Итог (глобальные): TPR = {tpr:.3f}, FDR = {fdr:.3f}")

if __name__ == "__main__":
    main()