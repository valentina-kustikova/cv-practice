import argparse
import cv2

from data_reader import DataReader
from base_detector import BaseDetector
from rcnn_resnet50 import FasterRcnnResNet50CocoDetector
from ssd_mobilenet_v2 import SsdMobilenetV2CocoDetector
from yolo_v4 import YoloV4CocoDetector
from metrics import compute_dataset_metrics

MODEL_CONFIGS = {
    "ssd_mobilenet_v2_coco": {
        "model": "models/mobilenet/frozen_inference_graph.pb",
        "config": "models/mobilenet/ssd_mobilenet_v2.pbtxt",
        "classes": "models/mobilenet/object_detection_classes_coco.txt",
    },
    "yolo_v4_coco": {
        "model": "models/yolov4/yolov4-tiny.weights",
        "config": "models/yolov4/yolov4.cfg",
        "classes": "models/yolov4/coco.names",
    },
    "faster_rcnn_resnet50_coco": {
        "model": "models/resnet50/frozen_inference_graph.pb",
        "config": "models/resnet50/faster_rcnn_resnet50_coco_2018_01_28.pbtxt",
        "classes": "models/resnet50/object_detection_classes_coco.txt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Обнаружение транспорта")

    parser.add_argument(
        "--images", required=True, help="Путь к директории с изображениями"
    )

    parser.add_argument("--ann", required=True, help="Путь к TXT файлу аннотаций")

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CONFIGS.keys(),
        help="Название модели детектора",
    )

    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")

    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")

    parser.add_argument(
        "--show", action="store_true", help="Показывать окна с детекциями"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold для матчинга детекций"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = MODEL_CONFIGS[args.model]

    detector = BaseDetector.create(
        args.model,
        model_path=cfg["model"],
        config_path=cfg["config"],
        classes_path=cfg["classes"],
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )

    reader = DataReader(
        images_dir=args.images,
        annotation_path=args.ann,
    )

    print("Кадров в датасете:", len(reader))

    all_gt = []
    all_pred = []

    for frame_id, image, gt_boxes in reader:
        detections = detector.detect(image)
        vis = detector.draw_detections(image, detections)

        all_gt.append(gt_boxes)
        all_pred.append(detections)

        if args.show:
            cv2.imshow("Обнаружение", vis)
            key = cv2.waitKey(1)
            if key == 27:  # Esc
                break

    cv2.destroyAllWindows()

    tpr, fdr, tp, fp, fn = compute_dataset_metrics(
        all_gt,
        all_pred,
        iou_threshold=args.iou,
    )

    print("\n=== Метрики по датасету ===")
    print(f"TPR (Recall): {tpr:.4f}")
    print(f"FDR: {fdr:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")


if __name__ == "__main__":
    main()
