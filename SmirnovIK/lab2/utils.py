import numpy as np

def read_annotations(filepath):
    annotations = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_id = int(parts[0])
            class_name = parts[1]

            xmin, ymin, xmax, ymax = map(int, parts[2:])
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]

            if frame_id not in annotations:
                annotations[frame_id] = []
            
            annotations[frame_id].append({
                'class': class_name,
                'bbox': bbox
            })
    return annotations

def class_color(cid):
    np.random.seed(cid)
    return tuple(int(x) for x in np.random.randint(0,255,3))

def load_class_names(path="classes.txt"):
    with open(path, "r") as f:
        return [c.strip() for c in f.readlines()]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]
    union = areaA + areaB - inter
    return inter / (union + 1e-9)
def evaluate_frame(gt_boxes, det_boxes, iou_thr=0.5):
    gt_matched = set()
    det_matched = set()
    det_boxes = sorted(det_boxes, key=lambda d: d[4], reverse=True)
    for gi, (gt_class, gt_box) in enumerate(gt_boxes):
        for di, (x,y,w,h,conf,det_class) in enumerate(det_boxes):
            if det_class.lower() == gt_class.lower():
                if iou(gt_box, (x,y,w,h)) >= iou_thr:
                    gt_matched.add(gi)
                    det_matched.add(di)
                    break

    TP = len(gt_matched)
    FP = len(det_boxes) - len(det_matched)
    FN = len(gt_boxes) - len(gt_matched)

    return  TP, FN, FP
