import cv2
import numpy as np
import argparse
import sys

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to an image',
                        type=str,
                        dest='input')
    parser.add_argument('-c', '--confidence_threshold',
                        help='Confidence threshold',
                        type=float,
                        default=0.5,
                        dest='conf_thr')
    parser.add_argument('-nms', '--nms_threshold',
                        help='NMS threshold',
                        type=float,
                        default=0.4,
                        dest='nms_thr')
    parser.add_argument('-s', '--input_shape',
                        type=str,
                        nargs=2,
                        default=[608, 608],
                        dest='input_shape')
    parser.add_argument('-w', '--weights',
                        help='Path to weights file',
                        dest='weights')
    parser.add_argument('-cfg', '--config',
                        help='Path to config file',
                        dest='config')
    parser.add_argument('-l', '--labels',
                        dest='labels')
    args = parser.parse_args()

    return args


class DetectorYOLOV4:

    def __init__(self, weights, config, names):
        self.net = cv2.dnn.readNet(weights, config)
        with open(names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame, conf_threshold=0.5, nms_threshold=0.4, input_size=(416, 416)):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, input_size, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, classes, confidences, class_ids = [], [], [], []
        detected_objects = {}

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                    width, height = int(detection[2] * w), int(detection[3] * h)
                    x, y = int(center_x - width / 2), int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    classes.append(class_id)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, width, height = boxes[i]
                detected_objects[self.classes[classes[i]]] = detected_objects.get(self.classes[classes[i]], 0) + 1
                color = self.colors[class_ids[i]]
                label = f"{self.classes[class_ids[i]]}: {confidences[i]:.3f}"
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detected_objects


def process_input(input_path, yolo, conf_threshold, nms_threshold, input_size):
    if input_path.endswith((".mp4", ".avi")):
        videoCapture = cv2.VideoCapture(input_path)
        frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = videoCapture.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('result.mp4', fourcc, fps / 4, (frame_width, frame_height))
        success, frame = videoCapture.read()
        count = 0
        while success:
            output_frame, detected_objects = yolo.detect_objects(frame, conf_threshold, nms_threshold, input_size)
            print('Objects in the frame:')
            for obj, count in detected_objects.items():
                print(f"{obj}: {count}")
            cv2.imshow('detection', output_frame)
            if cv2.waitKey(ord(' ')) > 0:
                break
            out.write(output_frame)
            success, frame = videoCapture.read()
            count += 1
        videoCapture.release()
        out.release()
    else:
        image = cv2.imread(input_path)
        if image is None:
            print("Error: Could not load the image.")
            return
        output_image, detected_objects = yolo.detect_objects(image, conf_threshold, nms_threshold, input_size)
        print('Objects in the frame:')
        for obj, count in detected_objects.items():
            print(f"{obj}: {count}")
        cv2.imshow("detection", output_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = cli_argument_parser()
    yolo = DetectorYOLOV4(args.weights, args.config, args.labels)
    process_input(args.input, yolo, args.conf_thr, args.nms_thr, tuple([int(i) for i in args.input_shape]))

if __name__ == '__main__':
    sys.exit(main() or 0)









