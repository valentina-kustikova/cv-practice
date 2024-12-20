import cv2 as cv
import numpy as np
import argparse
import sys

def load_model(config_path, weights_path):
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)
    return net

def detect_objects(image, net, conf_threshold, nms_threshold, classes):
    h, w = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "class_id": class_ids[i],
                "class_name": classes[class_ids[i]],
                "confidence": confidences[i],
                "box": boxes[i]
            })
    return results

def draw_detections(image, detections, colors):
    for det in detections:
        x, y, w, h = det['box']
        color = colors[det['class_id']]
        label = f"{det['class_name']}: {det['confidence']:.3f}"
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def process_frame(frame, net, conf_threshold, nms_threshold, classes, colors):
    height, width = frame.shape[:2]
    frame = cv.resize(frame, (width // 2, height // 2))
    detections = detect_objects(frame, net, conf_threshold, nms_threshold, classes)
    result_image = draw_detections(frame, detections, colors)
    cv.imshow("Detected Objects", result_image)
    stats = {}
    for det in detections:
        class_name = det['class_name']
        stats[class_name] = stats.get(class_name, 0) + 1

    print("Image Statistics:", stats)

def process_image(image_path, net, conf_threshold, nms_threshold, classes, colors):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    process_frame(image, net, conf_threshold, nms_threshold, classes, colors)
    cv.waitKey(0)
    cv.destroyAllWindows()

def process_video(video_path, net, conf_threshold, nms_threshold, classes, colors):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Unable to open video')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame, net, conf_threshold, nms_threshold, classes, colors)
        key = cv.waitKey(0) & 0xFF
        if key == ord('w'):
            continue
        if key == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            exit()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv4 and OpenCV DNN")
    parser.add_argument('-m', '--mode', 
                        help="Mode of operation: 'image' or 'video'. Default is 'image'.", 
                        type=str, 
                        dest='mode', 
                        default='image')
    parser.add_argument('-i', '--input', 
                        help="Path to input image or video file.", 
                        type=str, 
                        dest='input_path', 
                        required=True)
    parser.add_argument('-cl', '--classes', 
                        help="Path to the file containing class labels. Default is 'models/coco.names'.", 
                        type=str, 
                        dest='classes_path', 
                        default='models/coco.names')
    parser.add_argument('-c', '--config', 
                        help="Path to the YOLOv4 config file.", 
                        type=str, 
                        dest='config_path', 
                        default='models/yolov4.cfg')
    parser.add_argument('-w', '--weights', 
                        help="Path to the YOLOv4 weights file.", 
                        type=str, 
                        dest='weights_path', 
                        default='models/yolov4.weights')
    parser.add_argument('-ct', '--conf_threshold', 
                        help="Confidence threshold for detecting objects. Default is 0.5.", 
                        type=float, 
                        dest='conf_threshold', 
                        default=0.5)
    parser.add_argument('-nt', '--nms_threshold', 
                        help="Non-Maximum Suppression (NMS) threshold. Default is 0.4.", 
                        type=float, 
                        dest='nms_threshold', 
                        default=0.4)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    with open(args.classes_path, 'r') as f:
        CLASSES = f.read().strip().split('\n')
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = load_model(args.config_path, args.weights_path)
    
    if args.mode == 'image':
        process_image(args.input_path, net, args.conf_threshold, args.nms_threshold, CLASSES, COLORS)
    elif args.mode == 'video':
        process_video(args.input_path, net, args.conf_threshold, args.nms_threshold, CLASSES, COLORS)
    else:
        print("Invalid mode. Use 'image' or 'video'.")


if __name__ == '__main__':
    sys.exit(main() or 0)
