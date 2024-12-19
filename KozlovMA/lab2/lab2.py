import cv2
import numpy as np
import argparse
import os


class YOLOv4Detector:
    def __init__(self, weights_path, config_path, classes_path, conf_threshold=0.5, nms_threshold=0.6):
        self.net, self.layer_names = self.load_model(weights_path, config_path)
        self.classes = self.load_classes(classes_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    @staticmethod
    def load_classes(classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    @staticmethod
    def load_model(weights_path, config_path):
        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers

    def preprocess_image(self, image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        return height, width


    def detect_objects(self, image, height, width):
        outputs = self.net.forward(self.layer_names)
        detections = np.concatenate(outputs, axis=0)
        confidences = detections[:, 5:]
        class_ids = np.argmax(confidences, axis=1)
        confidence_scores = np.max(confidences, axis=1)


        valid_indices = confidence_scores > self.conf_threshold
        boxes = detections[:, :4]
        centers_x = (boxes[:, 0] * width).astype(int)
        centers_y = (boxes[:, 1] * height).astype(int)
        widths = (boxes[:, 2] * width).astype(int)
        heights = (boxes[:, 3] * height).astype(int)

        left_x = centers_x - widths // 2
        top_y = centers_y - heights // 2

        valid_boxes = np.column_stack((left_x[valid_indices], top_y[valid_indices], widths[valid_indices], heights[valid_indices]))
        valid_confidences = confidence_scores[valid_indices]
        valid_class_ids = class_ids[valid_indices]

        indexes = cv2.dnn.NMSBoxes(valid_boxes.tolist(), valid_confidences.tolist(), self.conf_threshold, self.nms_threshold)
        return valid_boxes, valid_confidences, valid_class_ids, indexes

class Visualizer:
    def __init__(self, classes):
        self.classes = classes
        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))

    def draw_detections(self, image, boxes, confidences, class_ids, indexes):
        detections = {}
        if (len(indexes)<=0): return
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            if class_ids[i] not in detections.keys():
                detections[class_ids[i]] = 0
            detections[class_ids[i]]+=1
            label = f"{self.classes[class_ids[i]]} {confidences[i]:.2f}"
            color = self.colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return detections
    @staticmethod
    def annotate_summary(image, detected_objects, classes):
        y_offset = 30
        if (len(detected_objects)<1): return
        for class_id, count in detected_objects.items():
            label = f"{classes[class_id]}: {count}"
            cv2.putText(image, label, (100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 30

    def render_image(self, image, boxes, confidences, class_ids, indexes):
        detected_objects = self.draw_detections(image, boxes, confidences, class_ids, indexes)
        self.annotate_summary(image, detected_objects, self.classes)
        return image

def process_image(input_path, output_path, detector, visualizer):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image could not be loaded.")
    
    height, width = detector.preprocess_image(image)
    boxes, confidences, class_ids, indexes = detector.detect_objects(image, height, width)
    result_image = visualizer.render_image(image, boxes, confidences, class_ids, indexes)

    cv2.imwrite(output_path, result_image)
    cv2.imshow("Detection Results", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(input_path, output_dir, detector, visualizer):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = detector.preprocess_image(frame)
        boxes, confidences, class_ids, indexes = detector.detect_objects(frame, height, width)
        result_frame = visualizer.render_image(frame, boxes, confidences, class_ids, indexes)
        frames.append(result_frame)
        cv2.imshow("Detection Results", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_video_path = os.path.join(output_dir, "output_video.avi")
    create_video(frames, output_video_path)

def create_video(frames, output_path, fps=60):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["image", "video"], required=True, help="Processing mode (image or video)")
    parser.add_argument("-i", "--input", required=True, help="Input file path (image or video)")
    parser.add_argument("-o", "--output", default="output.jpg", help="Output file path")
    parser.add_argument("-d", "--output_dir", default="output_frames", help="Output directory for video processing")
    parser.add_argument("-c", "--config", default="yolov4.cfg", help="YOLOv4 config file path")
    parser.add_argument("-w", "--weights", default="yolov4.weights", help="YOLOv4 weights file path")
    parser.add_argument("-cl", "--classes", default="coco.names", help="Path to classes file")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.4, help="NMS threshold")
    return parser.parse_args()

def main():
    args = parse_arguments()
    detector = YOLOv4Detector(args.weights, args.config, args.classes, args.conf_threshold, args.nms_threshold)
    visualizer = Visualizer(detector.classes)

    if args.mode == "image":
        process_image(args.input, args.output, detector, visualizer)
    elif args.mode == "video":
        process_video(args.input, args.output_dir, detector, visualizer)

if __name__ == "__main__":
    main()
