import cv2 as cv
import numpy as np
import argparse
import sys

import time

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
    return cv.dnn.readNet(config_path, weights_path)

def GetImage(image_path):
    # ImageProcessing(image_path, net, conf_threshold, nms_threshold, classes, colors)
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Unable to load image')
    return image

def GetCap(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Unable to open video')
    return cap

def VideoProcessing(video_path, YOLO, conf_threshold, nms_threshold, classes, colors):
    cap = GetCap(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ProcessFrame(frame, YOLO, conf_threshold, nms_threshold, classes, colors)

        key = cv.waitKey(0)

        if key == ord('v'):
            break

def ImageProcessing(image_path, net, conf_threshold, nms_threshold, classes, colors):
    image = GetImage(image_path)
    ProcessFrame(image, net, conf_threshold, nms_threshold, classes, colors)
    cv.waitKey(0)

def ProcessFrame(frame, YOLO, conf_threshold, nms_threshold, classes, colors):  # This function is a kind of control: it calls the detection function and calls the render function
    detections = Detect(frame, YOLO, conf_threshold, nms_threshold, classes)
    result_image = Draw(frame, detections, colors)
    cv.imshow("Detected Objects", result_image)
    stats = {}
    for det in detections:
        class_name = det['class_name']
        stats[class_name] = stats.get(class_name, 0) + 1

    print("Image Statistics:", stats)


def Detect(image, YOLO, conf_threshold, nms_threshold, classes):
    h, w = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

    # print(image.shape)
    # print(blob.shape)

    YOLO.setInput(blob)

    layers = YOLO.getLayerNames()  # Get the name of all layers of the network.
    out_layers = [layers[i - 1] for i in YOLO.getUnconnectedOutLayers()]  # Get the index of the output layers (327 353 379)

    # прямая связь (вывод) и получение выхода сети
    start = time.perf_counter()
    layer_outputs = YOLO.forward(out_layers)  # for example [0.9295312 , 0.9318892 , 0.20397013, ..., 0. , 0. ,0. ] - the probability of detecting objects (coco.names)
    time_took = time.perf_counter() - start
    print(f"took: {time_took:.2f}s")
    boxes, probability_list, founded_classes = [], [], []

    for output in layer_outputs:
    # перебираем каждое обнаружение объекта
        for detection in output:
            # извлекаем идентификатор класса (метку) и достоверность (как вероятность) обнаружения текущего объекта
            scores = detection[5:]
            class_id = np.argmax(scores)
            probability = scores[class_id]
            # отбросьте слабые прогнозы, убедившись, что обнаруженные
            # вероятность больше минимальной вероятности
            if probability > conf_threshold:
                # масштабируем координаты ограничивающего прямоугольника относительно
                # размер изображения, т.к "Юла" на самом деле
                # возвращает центральные координаты (x, y) ограничивающего
                # поля, за которым следуют ширина и высота поля
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # используем центральные координаты (x, y) для получения вершины и
                # и левый угол ограничительной рамки
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # обновим  список координат ограничивающего прямоугольника, достоверности,
                # и идентификаторы класса
                boxes.append([x, y, int(width), int(height)])
                probability_list.append(float(probability))
                founded_classes.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, probability_list, conf_threshold, nms_threshold)

    results = []

    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "class_id": founded_classes[i],
                "class_name": classes[founded_classes[i]],
                "confidence": probability_list[i],
                "box": boxes[i]
            })
    
    print(results)

    return results

def Draw(image, detections, colors):
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


def main():
    args = argument_parser()

    # get data from command line

    with open(args.classes_path, 'r') as f:
        CLASSES = f.read().strip().split('\n')

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    YOLO = Model(args.config_path, args.weights_path)
    if args.mode == 'video':    
        VideoProcessing(args.input_path, YOLO, args.conf_threshold, args.nms_threshold, CLASSES, COLORS)
    elif args.mode == 'image':
        ImageProcessing(args.input_path, YOLO, args.conf_threshold, args.nms_threshold, CLASSES, COLORS)

if __name__ == '__main__':
    sys.exit(main() or 0)