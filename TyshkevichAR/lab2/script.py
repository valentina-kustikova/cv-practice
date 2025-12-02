import cv2
import os
import argparse
import numpy as np


def parse_gt(txt_path):
    gt = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            frame_id = int(parts[0])
            cls = parts[1].lower()
            x1, y1, x2, y2 = map(int, parts[2:])
            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append({'class': cls, 'box': [x1, y1, x2, y2]})
    return gt


class MetricsEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.iou_threshold = iou_threshold

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou_val = interArea / (boxAArea + boxBArea - interArea + 1e-6)
        return iou_val

    def update(self, dets, gts):
        dets.sort(key=lambda x: x['conf'], reverse=True)
        matched_gt = set()
        tp = 0
        for det in dets:
            det_box = det['box']
            det_class = det['class']
            max_iou = 0
            match_idx = -1
            for idx, g in enumerate(gts):
                if idx in matched_gt:
                    continue
                if g['class'] != det_class:
                    continue
                cur_iou = self.iou(det_box, g['box'])
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    match_idx = idx
            if max_iou > self.iou_threshold and match_idx != -1:
                tp += 1
                matched_gt.add(match_idx)
        fp = len(dets) - tp
        fn = len(gts) - tp
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

    def compute(self):
        tpr = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        fdr = self.total_fp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        return tpr, fdr


class ObjectDetector:
    def __init__(self, model_path, config_path, classes_path, framework):
        if framework == 'darknet':
            self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        elif framework == 'caffe':
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            self.classes = [
                '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                'sofa', 'train', 'tvmonitor'
            ]

        self.vehicle_classes = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
        self.vehicle_indices = []
        for i, c in enumerate(self.classes):
            class_lower = c.lower()
            if any(vehicle in class_lower for vehicle in ['bicycle', 'car', 'motorcycle', 'motorbike',
                                                          'airplane', 'aeroplane', 'bus', 'train', 'truck', 'boat']):
                self.vehicle_indices.append(i)

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def detect(self, image):
        height, width = image.shape[:2]
        blob = self.preprocess(image)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outs = self.net.forward(layer_names)
        boxes, confs, class_ids = self.postprocess(outs, (width, height))
        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        dets = []
        for i in indices:
            if class_ids[i] in self.vehicle_indices:
                box = boxes[i]
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[0] + box[2])
                y2 = int(box[1] + box[3])
                dets.append({
                    'class': self.classes[class_ids[i]],
                    'conf': confs[i],
                    'box': [x1, y1, x2, y2]
                })
        return dets


class YOLODetector(ObjectDetector):
    def __init__(self, model_path, config_path, classes_path):
        super().__init__(model_path, config_path, classes_path, 'darknet')
        self.input_size = 416

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)

    def postprocess(self, outs, shape):
        boxes = []
        confs = []
        class_ids = []
        width, height = shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id] * detection[4]
                if conf > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        return boxes, confs, class_ids


class SSDDetector(ObjectDetector):
    def __init__(self, model_path, config_path, classes_path):
        super().__init__(model_path, config_path, classes_path, 'caffe')
        self.input_size = 300

    def preprocess(self, image):
        return cv2.dnn.blobFromImage(image, 0.007843, (self.input_size, self.input_size),
                                     (127.5, 127.5, 127.5), swapRB=False)

    def postprocess(self, outs, shape):
        boxes = []
        confs = []
        class_ids = []
        width, height = shape

        detections = outs[0] if isinstance(outs, tuple) else outs

        for i in range(detections.shape[2]):
            detection = detections[0, 0, i]
            confidence = detection[2]

            if confidence > 0.5:
                class_id = int(detection[1])

                if class_id == 0:
                    continue

                if class_id >= len(self.classes):
                    continue

                x_left = int(detection[3] * width)
                y_top = int(detection[4] * height)
                x_right = int(detection[5] * width)
                y_bottom = int(detection[6] * height)

                if (x_right > x_left and y_bottom > y_top and
                        x_left >= 0 and y_top >= 0 and
                        x_right <= width and y_bottom <= height):
                    w = x_right - x_left
                    h = y_bottom - y_top

                    boxes.append([x_left, y_top, w, h])
                    confs.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confs, class_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default='data/imgs_MOV03478', help='Path to the directory with images')
    parser.add_argument('--gt_txt', default='data/mov03478.txt', help='Path to the ground truth text file')
    parser.add_argument('--model', required=True, choices=['yolo', 'ssd'], help='Model to use: yolo or ssd')
    parser.add_argument('--model_path', required=True, help='Path to the model weights file')
    parser.add_argument('--config_path', required=True, help='Path to the model config file')
    parser.add_argument('--classes_path', default='coco.names', help='Path to the classes names file')
    parser.add_argument('--display', action='store_true', help='Display frames with detections')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for display (default: 30)')
    args = parser.parse_args()

    if args.model == 'ssd' and not os.path.exists(args.classes_path):
        os.makedirs(os.path.dirname(args.classes_path), exist_ok=True)
        ssd_classes = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
        with open(args.classes_path, 'w') as f:
            for cls in ssd_classes:
                f.write(cls + '\n')

    if args.model == 'yolo':
        detector = YOLODetector(args.model_path, args.config_path, args.classes_path)
    elif args.model == 'ssd':
        detector = SSDDetector(args.model_path, args.config_path, args.classes_path)

    gt = parse_gt(args.gt_txt)
    image_files = sorted([f for f in os.listdir(args.images_dir) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))
    evaluator = MetricsEvaluator()

    for img_file in image_files:
        frame_id = int(os.path.splitext(img_file)[0])
        image_path = os.path.join(args.images_dir, img_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        dets = detector.detect(image)
        gts = gt.get(frame_id, [])
        evaluator.update(dets, gts)

        if args.display:
            for det in dets:
                box = det['box']
                cls = det['class']
                conf = det['conf']
                color = (0, 255, 0)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                text = f"{cls} {conf:.3f}"
                cv2.putText(image, text, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for g in gts:
                box = g['box']
                color = (255, 0, 0)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
                cv2.putText(image, g['class'], (box[0], box[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Press Q for Exit", image)
            delay = int(1000 / args.fps)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    tpr, fdr = evaluator.compute()
    print(f"TPR: {tpr:.3f}")
    print(f"FDR: {fdr:.3f}")