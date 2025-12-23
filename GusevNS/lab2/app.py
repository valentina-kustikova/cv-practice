import argparse
from pathlib import Path
import cv2
from detectors import MODEL_REGISTRY
from utils.dataset import load_annotations, list_image_paths, load_frame
from utils.metrics import match_detections, compute_metrics
from utils.visualization import draw_detections, draw_truths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="Data/imgs_MOV03478")
    parser.add_argument("--annotations", type=str, default="Data/mov03478.txt")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), default="yolov4_tiny")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = load_annotations(args.annotations)
    image_paths = list_image_paths(args.images)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    detector_cls = MODEL_REGISTRY[args.model]
    detector = detector_cls(model_dir=Path(args.models_dir) / args.model)
    aggregated = {"tp": 0, "fp": 0, "fn": 0}
    for index, image_path in enumerate(image_paths):
        frame_id = int(image_path.stem)
        image = load_frame(image_path)
        if image is None:
            continue
        detections = detector.detect(image)
        truths = annotations.get(frame_id, [])
        tp, fp, fn = match_detections(detections, truths, args.iou)
        aggregated["tp"] += tp
        aggregated["fp"] += fp
        aggregated["fn"] += fn
        if args.display:
            frame = draw_detections(image, detections)
            frame = draw_truths(frame, truths)
            cv2.imshow("detections", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
    tpr, fdr = compute_metrics(aggregated)
    print(f"Processed frames: {len(image_paths)}")
    print(f"True positives: {aggregated['tp']}")
    print(f"False positives: {aggregated['fp']}")
    print(f"False negatives: {aggregated['fn']}")
    print(f"TPR: {tpr:.3f}")
    print(f"FDR: {fdr:.3f}")
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

