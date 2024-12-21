import cv2 as cv
import argparse
import numpy as np
import sys

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='Path to an image or video',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='out.jpg',
                        dest='out_image_path')
    parser.add_argument('-m', '--mode',
                        help='Mode (image, video)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-cfg', '--config',
                        help='Path to the cfg file',
                        type=str,
                        default='yolov3.cfg',
                        dest='config_path')
    parser.add_argument('-w', '--weights',
                        help='Path to the weights file',
                        type=str,
                        dest='weights_path',
                        default='yolov3.weights')
    parser.add_argument('-names', '--coco.names',
                        help='Path to the coco.names file',
                        type=str, 
                        dest='names_path',
                        default='coco.names')
    parser.add_argument('-ct', '--conf_threshold',
                        help='Confidence threshold',
                        type=float,
                        dest='confidence_threshold',
                        default=0.5)
    parser.add_argument('-nt', '--nms_threshold',
                        help='NMS threshold',
                        type=float,
                        dest='nms_threshold',
                        default=0.4)
    
    args = parser.parse_args()

    return args

def load_names(names_path):
    with open(names_path, "r") as f:
        names = [line.strip() for line in f.readlines()]
    return names

def read_img(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    src_image = cv.imread(image_path)
    return src_image

def show_img(image):
    if image is None:
        raise ValueError('Empty path to the image')
    cv.imshow('YOLO', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preprocess_image(net, image):
    height, width = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    return height, width

def detect_objects(net, names, image, height, width, confidence_threshold, nms_threshold, colors):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    detected_objects = []
    for i in indices:
        x, y, w, h = boxes[i]
        label = str(names[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.putText(image, f"{label}: {confidence:.3f}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detected_objects.append({"class": label, "confidence": confidence, "bbox": (x, y, w, h)})
    return detected_objects, image

def count_objects(detected_objects):
    counts = {}
    for obj in detected_objects:
        class_name = obj['class']
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts

def output_objects(image, detected_objects):
    counts = count_objects(detected_objects)
    y_offset = 20
    for class_name, count in counts.items():
        text = f"{class_name}: {count}"
        cv.putText(image, text, (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        y_offset += 20
    return image

def process_video(video_path, net, names, confidence_threshold, nms_threshold, colors):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = preprocess_image(net, frame)
        detected_objects, frame = detect_objects(net, names, frame, height, width, confidence_threshold, nms_threshold, colors)
        output_frame = output_objects(frame, detected_objects)
        cv.imshow('YOLO', output_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def main():
    args = cli_argument_parser()
    net = cv.dnn.readNetFromDarknet(args.config_path, args.weights_path)
    names = load_names(args.names_path)
    colors = np.random.uniform(0, 255, size=(len(names), 3))
    
    if args.mode == 'image':
        image = read_img(args.image_path)
        height, width = preprocess_image(net, image)
        detected_objects, image2 = detect_objects(net, names, image, height, width, args.confidence_threshold, args.nms_threshold, colors)
        output_image = output_objects(image2, detected_objects)
        show_img(output_image)
    elif args.mode == 'video':
        process_video(args.image_path, net, names, args.confidence_threshold, args.nms_threshold, colors)
    else:
        raise ValueError('Unsupported mode')


if __name__ == '__main__':
    sys.exit(main() or 0)
