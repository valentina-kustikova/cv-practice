import cv2 as cv
import numpy as np
import argparse

class YOLODetector:
    def __init__(self, weights, config, names):
        self.net = cv.dnn.readNet(weights, config)
        self.classes = self.load_classes(names)
        self.colors = self.generate_colors()
        self.output_layers = self.get_output_layers()

    def load_classes(self, names):
        with open(names, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def generate_colors(self):
        return np.random.uniform(0, 255, size=(len(self.classes), 3))

    def get_output_layers(self):
        return [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame, conf_threshold=0.5, nms_threshold=0.4, input_size=(416, 416)):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 0.00392, input_size, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    box = self.extract_box(detection, w, h)
                    boxes.append(box)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        self.draw_boxes(frame, boxes, confidences, class_ids, indices)

        return frame

    def extract_box(self, detection, width, height):
        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
        box_width, box_height = int(detection[2] * width), int(detection[3] * height)
        x = int(center_x - box_width / 2)
        y = int(center_y - box_height / 2)
        return [x, y, box_width, box_height]

    def draw_boxes(self, frame, boxes, confidences, class_ids, indices):
        for i in indices.flatten():
            x, y, width, height = boxes[i]
            color = self.colors[class_ids[i]]
            label = f"{self.classes[class_ids[i]]}: {confidences[i]:.3f}"
            cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_frame(self, frame, conf_threshold, nms_threshold, input_size):
        output_frame = self.detect_objects(frame, conf_threshold, nms_threshold, input_size)
        return output_frame

def display_frame(frame, window_name="Object Detection", wait_for_key=False):
    cv.imshow(window_name, frame)
    return cv.waitKey(0) if wait_for_key else cv.waitKey(1) != ord("q")

def cleanup():
    cv.destroyAllWindows()

def process_input(input_path, yolo, conf_threshold, nms_threshold, input_size):
    if input_path.endswith((".mp4", ".avi")):
        cap = cv.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_frame = yolo.process_frame(frame, conf_threshold, nms_threshold, input_size)
            if not display_frame(output_frame):
                break
        cap.release()
    else:
        image = cv.imread(input_path)
        if image is None:
            print("Error: Could not load the image.")
            return
        output_image = yolo.process_frame(image, conf_threshold, nms_threshold, input_size)
        display_frame(output_image, wait_for_key=True)
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Object Detection with YOLO")
    parser.add_argument('-i', '--input', required=True, help="Path to the input image or video")
    parser.add_argument('-c', '--confidence_threshold', type=float, default=0.5, help="Confidence threshold ( default: 0.5)")
    parser.add_argument('-n', '--nms_threshold', type=float, default=0.4, help="NMS threshold (default: 0.4)")
    parser.add_argument('-s', '--input_size', type=int, nargs=2, default=[416, 416], help="YOLO input size (default: 416 416)")
    parser.add_argument('-w', '--weights', required=True, help="Path to YOLO weights file")
    parser.add_argument('-cfg', '--config', required=True, help="Path to YOLO config file")
    parser.add_argument('-names', '--names', required=True, help="Path to class names file")

    args = parser.parse_args()
    yolo = YOLODetector(args.weights, args.config, args.names)
    process_input(args.input, yolo, args.confidence_threshold, args.nms_threshold, tuple(args.input_size))

if __name__ == '__main__':
    main()