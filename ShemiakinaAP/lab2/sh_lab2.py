#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np
import argparse
import sys

def argument_parser():
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv4 and OpenCV DNN")
    parser.add_argument('-m', '--mode', 
                        help="Mode of operation: 'image' or 'video'. Default is 'image'.", 
                        type=str, 
                        dest='mode', 
                        default='image')
    parser.add_argument('-i', '--input', 
                        help="Path to an image or a video file.", 
                        type=str, 
                        dest='input_path', 
                        required=True)
    parser.add_argument('-cl', '--classes', 
                        help="Path to the file containing class labels. Default is 'coco.names'.", 
                        type=str, 
                        dest='classes_path', 
                        default='coco.names')
    parser.add_argument('-c', '--config', 
                        help="Path to the YOLOv4 config file. Default is 'yolov4.cfg'", 
                        type=str, 
                        dest='config_path', 
                        default='yolov4.cfg')
    parser.add_argument('-w', '--weights', 
                        help="Path to the YOLOv4 weights file. Default is 'yolov4.weights'", 
                        type=str, 
                        dest='weights_path', 
                        default='yolov4.weights')
    parser.add_argument('-ct', '--conf_threshold', 
                        help="Confidence threshold for detecting objects. Default is 0.5.", 
                        type=float, 
                        dest='conf_threshold', 
                        default=0.5)
    parser.add_argument('-nt', '--nms_threshold', 
                        help="NMS threshold. Default is 0.4.", 
                        type=float, 
                        dest='nms_threshold', 
                        default=0.4)
    args = parser.parse_args()
    return args

def Model(config_path, weights_path):
    net = cv.dnn.readNet(config_path, weights_path)
    return net

def DetectObjects(image, net, conf_threshold, nms_threshold, classes):
    h, w = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
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

def DrawDetections(image, detections, colors):
    for det in detections:
        x, y, w, h = det['box']
        color = colors[det['class_id']]
        label = f"{det['class_name']}: {det['confidence']:.3f}"
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_y = max(y - 10, 10)
        cv.rectangle(image, (x, label_y - label_size[1] - 5), (x + label_size[0], label_y), color, cv.FILLED)
        
        cv.putText(image, label, (x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image

def ProcessFrame(frame, net, conf_threshold, nms_threshold, classes, colors):
    detections = DetectObjects(frame, net, conf_threshold, nms_threshold, classes)
    result_image = DrawDetections(frame, detections, colors)
    cv.imshow("Detected Objects", result_image)
    stats = {}
    for det in detections:
        class_name = det['class_name']
        stats[class_name] = stats.get(class_name, 0) + 1

    print("Image Statistics:", stats)

def ImageProcessing(image_path, net, conf_threshold, nms_threshold, classes, colors):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    ProcessFrame(image, net, conf_threshold, nms_threshold, classes, colors)
    cv.waitKey(0)
    cv.destroyAllWindows()

def VideoProcessing(video_path, net, conf_threshold, nms_threshold, classes, colors):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Unable to open video')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ProcessFrame(frame, net, conf_threshold, nms_threshold, classes, colors)
        key = cv.waitKey(0) & 0xFF
        if key == ord('f'):
            continue
        if key == ord('v'):
            break

def main():
    args = argument_parser()

    with open(args.classes_path, 'r') as f:
        CLASSES = f.read().strip().split('\n')
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = Model(args.config_path, args.weights_path)

    if args.mode == 'image':
        ImageProcessing(args.input_path, net, args.conf_threshold, args.nms_threshold, CLASSES, COLORS)
    elif args.mode == 'video':
        VideoProcessing(args.input_path, net, args.conf_threshold, args.nms_threshold, CLASSES, COLORS)
    else:
        print("Invalid mode. Use 'image' or 'video'.")


if __name__ == '__main__':
    sys.exit(main() or 0)

