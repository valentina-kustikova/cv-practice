import cv2
import numpy as np
import argparse

# Загрузка предобученной модели YOLOv3
def initialize_yolo(model_cfg="yolov3.cfg", model_weights="yolov3.weights", class_names_file="coco.names"):
    net = cv2.dnn.readNet(model_weights, model_cfg)
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    output_layer_names = [layer_names[i - 1] for i in output_layers]
    classes = []
    with open(class_names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layer_names, classes

# Подготовка глобальных переменных
net, output_layer_names, classes = initialize_yolo()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Функция для загрузки изображения
def load_image(image_path):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open/read the image file: {image_path}")
            return None
        return image
    return image_path

# Функция для предварительной обработки изображения
def preprocess_image(image, scale=0.00392, size=(416, 416)):
    return cv2.dnn.blobFromImage(image, scale, size, (0, 0, 0), True, crop=False)

# Функция для анализа выходов нейронной сети
def process_network_output(outs, height, width, confidence_threshold):
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
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

    return class_ids, confidences, boxes

# Функция для применения немаксимального подавления
def apply_non_max_suppression(boxes, confidences, nms_threshold):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)
    return indexes

# Функция для отрисовки результатов
def draw_detections(image, boxes, confidences, class_ids, indexes):
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]].tolist()  # Цвет для данного класса
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidences[i]:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Основная функция для детектирования объектов на изображении
def detect_objects_on_image(image, confidence_threshold=0.5, nms_threshold=0.4):
    # Загрузка изображения
    image = load_image(image)
    if image is None:
        return None

    height, width, channels = image.shape

    # Подготовка изображения для нейронной сети
    blob = preprocess_image(image)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Анализ выходов нейронной сети
    class_ids, confidences, boxes = process_network_output(outs, height, width, confidence_threshold)

    # Применение немаксимального подавления
    indexes = apply_non_max_suppression(boxes, confidences, nms_threshold)

    # Отрисовка результатов
    image = draw_detections(image, boxes, confidences, class_ids, indexes)

    return image

# Функция для детектирования объектов на видео
def detect_objects_on_video(video_path, confidence_threshold=0.5, nms_threshold=0.4):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0  # Счётчик кадров

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        frame_count += 1

        # Обрабатываем только каждый пятый кадр
        if frame_count % 5 == 0:
            frame = detect_objects_on_image(frame, confidence_threshold, nms_threshold)

            cv2.imshow("Video", frame)

        # Проверка нажатия клавиши для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection using YOLOv3 and OpenCV.")
    parser.add_argument("--image", help="Path to the image file.")
    parser.add_argument("--video", help="Path to the video file.")
    parser.add_argument("-c", "--confidence", type=float, help="Confidence threshold for object detection", default=0.5)
    parser.add_argument("-n", "--nms", type=float, help="Non-maximum suppression threshold", default=0.4)
    args = parser.parse_args()

    if args.image:
        # Детектирование объектов на изображении
        processed_image = detect_objects_on_image(args.image, args.confidence, args.nms)
        if processed_image is not None:
            cv2.imshow("Detected Image", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif args.video:
        # Детектирование объектов на видео
        detect_objects_on_video(args.video, args.confidence, args.nms)
    else:
        print("Error: Please provide an image or video file path using --image or --video.")