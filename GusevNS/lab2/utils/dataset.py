from pathlib import Path
import cv2


class GroundTruth:
    def __init__(self, class_name, box):
        self.class_name = class_name
        self.box = box


def load_annotations(path):
    annotations = {}
    path = Path(path)
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            frame_id = int(parts[0])
            class_name = parts[1].lower()
            x1, y1, x2, y2 = map(int, parts[2:])
            box = (x1, y1, x2, y2)
            annotations.setdefault(frame_id, []).append(GroundTruth(class_name, box))
    return annotations


def list_image_paths(folder):
    folder = Path(folder)
    return sorted([path for path in folder.glob("*.jpg")])


def load_frame(path):
    return cv2.imread(str(path))

