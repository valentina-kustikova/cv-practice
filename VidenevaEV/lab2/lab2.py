#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np
import argparse
import sys

def initialize_yolo_model(config_path, weights_path):
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)
    return net

def extract_objects_from_image(img, model, threshold, nms_threshold, class_labels):
    h, w = img.shape[:2]
    blob = cv.dnn.blobFromImage(img, scalefactor=1/255.0, size=(608, 608), mean=(0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)

    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(output_layers)

    object_classes, confidences, bounding_boxes = [], [], []
    for output in outputs:
        for detection in output:
        
            scores = detection[5:]
            
            class_id = np.argmax(scores)
            
            confidence = scores[class_id]
            if confidence > threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, obj_width, obj_height) = box.astype("int")

                x = int(center_x - (obj_width / 2))
                y = int(center_y - (obj_height / 2))
                bounding_boxes.append([x, y, int(obj_width), int(obj_height)])
                confidences.append(float(confidence))
                object_classes.append(class_id)

    indices = cv.dnn.NMSBoxes(bounding_boxes, confidences, threshold, nms_threshold)
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            detected_objects.append({
                "class_id": object_classes[i],
                "class_name": class_labels[object_classes[i]],
                "confidence": confidences[i],
                "box": bounding_boxes[i]
            })
    return detected_objects

def mark_detected_objects(image, detected_objects, colors):
    for obj in detected_objects:
        x, y, w, h = obj['box']
        color = colors[obj['class_id']]
        label = f"{obj['class_name']}: {obj['confidence']:.3f}"
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def handle_image(image, model, threshold, nms_threshold, labels, color_map):
    height, width = image.shape[:2]
    image_resized = cv.resize(image, (width // 2, height // 2))
    detected_items = extract_objects_from_image(image_resized, model, threshold, nms_threshold, labels)
    annotated_image = mark_detected_objects(image_resized, detected_items, color_map)
    cv.imshow("Detected Objects", annotated_image)

    object_counts = {}
    for item in detected_items:
        label = item['class_name']
        object_counts[label] = object_counts.get(label, 0) + 1

    print("Object Count:", object_counts)

def handle_input_image(image_path, model, threshold, nms_threshold, labels, color_map):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError("Unable to load image")
    handle_image(img, model, threshold, nms_threshold, labels, color_map)
    cv.waitKey(0)
    cv.destroyAllWindows()

def handle_video(video_path, model, threshold, nms_threshold, labels, color_map):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        handle_image(frame, model, threshold, nms_threshold, labels, color_map)
        key = cv.waitKey(1) & 0xFF
        if key == ord('a'):
            cap.release()
            cv.destroyAllWindows()
            exit()

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Detect objects in images or video using YOLOv4")
    parser.add_argument('-m', '--mode', help="Mode of operation: 'image' or 'video'. Default is 'image'.", type=str, dest='mode', default='image')
    parser.add_argument('-i', '--input', help="Path to input image or video file.", type=str, dest='input_path', required=True)
    parser.add_argument('-cl', '--classes', help="Path to the file with class names (default: 'models/coco.names').", type=str, dest='classes_path', default='models/coco.names')
    parser.add_argument('-c', '--config', help="Path to YOLOv4 config file.", type=str, dest='config_path', default='models/yolov4.cfg')
    parser.add_argument('-w', '--weights', help="Path to YOLOv4 weights file.", type=str, dest='weights_path', default='models/yolov4.weights')
    parser.add_argument('-ct', '--conf_threshold', help="Confidence threshold for detection. Default is 0.5.", type=float, dest='conf_threshold', default=0.3) #пороги для уверенности
    parser.add_argument('-nt', '--nms_threshold', help="NMS threshold. Default is 0.4.", type=float, dest='nms_threshold', default=0.4)
    return parser.parse_args()

def main():
    args = parse_command_line_arguments()

    with open(args.classes_path, 'r') as f:
        class_labels = f.read().strip().split('\n')
    color_map = np.random.uniform(0, 255, size=(len(class_labels), 3))
    model = initialize_yolo_model(args.config_path, args.weights_path)

    if args.mode == 'image':
        handle_input_image(args.input_path, model, args.conf_threshold, args.nms_threshold, class_labels, color_map)
    elif args.mode == 'video':
        handle_video(args.input_path, model, args.conf_threshold, args.nms_threshold, class_labels, color_map)
    else:
        print("Invalid mode. Please use 'image' or 'video'.")

if __name__ == '__main__':
    sys.exit(main() or 0)

