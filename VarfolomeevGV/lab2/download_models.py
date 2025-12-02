import sys
import tarfile
import urllib.request
from pathlib import Path


BASE_DIR = Path("models")
YOLO_DIR = BASE_DIR / "yolo"
FRCNN_DIR = BASE_DIR / "faster_rcnn"
RETINANET_DIR = BASE_DIR / "retinanet"

COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
YOLO_ONNX_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx"
FRCNN_TAR_URL = "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz"
FRCNN_PBTXT_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"

RETINANET_ONNX_URL = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx"

SUPPORT_LINKS = [
    ("YOLOv5", "https://github.com/ultralytics/yolov5"),
    ("RetinaNet ONNX", "https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/retinanet"),
    ("ONNX Zoo", "https://github.com/onnx/models"),
]


def download(label: str, url: str, destination: Path, failures: list) -> bool:
    if destination.exists():
        print(f"[OK] {label}")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)

    def hook(block_idx, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_idx * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\r  {label}: {percent:5.1f}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, hook)
        sys.stdout.write("\n")
        print(f"  {destination.name} готов")
        return True
    except Exception as exc:
        sys.stdout.write("\n")
        failures.append((label, url, str(exc)))
        print(f"  Не удалось: {exc}")
        return False


def cleanup_yolo_directory():
    allowed = {"yolov5s.pt", "yolov5s.onnx"}
    for path in YOLO_DIR.glob("yolov5*"):
        if path.name not in allowed and path.is_file():
            path.unlink()
            print(f"  Удалён лишний файл {path.name}")


def ensure_frcnn_weights(failures: list):
    tar_path = FRCNN_DIR / "faster_rcnn_inception_v2_coco_2018_01_28.tar.gz"
    graph_path = FRCNN_DIR / "frozen_inference_graph.pb"

    if not graph_path.exists():
        if download("Faster R-CNN archive", FRCNN_TAR_URL, tar_path, failures):
            try:
                with tarfile.open(tar_path, "r:gz") as archive:
                    archive.extractall(FRCNN_DIR)
                extracted = FRCNN_DIR / "faster_rcnn_inception_v2_coco_2018_01_28" / "frozen_inference_graph.pb"
                if extracted.exists():
                    extracted.rename(graph_path)
                    print(f"  {graph_path.name} готов")
            except Exception as exc:
                failures.append(("Faster R-CNN unpack", FRCNN_TAR_URL, str(exc)))
                print(f"  Ошибка распаковки: {exc}")
    else:
        print("[OK] Faster R-CNN graph")

    pbtxt_path = FRCNN_DIR / "faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    download("Faster R-CNN config", FRCNN_PBTXT_URL, pbtxt_path, failures)


def ensure_retinanet(failures: list):
    """
    Скачивание RetinaNet модели.
    Используем ONNX модель из официального ONNX Model Zoo.
    """
    target = RETINANET_DIR / "retinanet-9.onnx"
    
    if not target.exists():
        print("Скачивание RetinaNet ONNX...")
        download("RetinaNet-9.onnx", RETINANET_ONNX_URL, target, failures)
    else:
        print("[OK] RetinaNet ONNX")


def ensure_yolo(failures: list):
    download("YOLOv5s.onnx", YOLO_ONNX_URL, YOLO_DIR / "yolov5s.onnx", failures)
    cleanup_yolo_directory()


def main():
    BASE_DIR.mkdir(exist_ok=True)
    YOLO_DIR.mkdir(exist_ok=True)
    FRCNN_DIR.mkdir(exist_ok=True)
    RETINANET_DIR.mkdir(exist_ok=True)

    failures = []
    print("=== Скачивание моделей ===")

    download("COCO labels", COCO_NAMES_URL, BASE_DIR / "coco.names", failures)
    ensure_yolo(failures)
    ensure_frcnn_weights(failures)
    ensure_retinanet(failures)

    if failures:
        print("\nНе всё удалось загрузить автоматически. Проверьте ссылки ниже и скачайте модели вручную (форматы .pt или .onnx подойдут, при необходимости используйте конвертер Ultralytics).")
        for label, url, reason in failures:
            print(f" - {label}: {url} ({reason})")
        print("\nПолезные репозитории:")
        for name, url in SUPPORT_LINKS:
            print(f"   {name}: {url}")
    else:
        print("\nВсе модели скачаны и разложены по папкам.")


if __name__ == "__main__":
    main()

