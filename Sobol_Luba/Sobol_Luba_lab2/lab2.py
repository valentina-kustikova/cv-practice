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

def preprocess_image(image, scale=0.00392, size=(416, 416)):
    return cv2.dnn.blobFromImage(image, scale, size, (0, 0, 0), True, crop=False)

# Функция для детектирования объектов на изображении или кадре
def detect_objects_on_image(image):
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            print(f"Error: Could not open/read the image file: {image}")
            return None
    
    height, width, channels = image.shape

    # Подготовка изображения для нейронной сети
    blob = preprocess_image(image)    
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Инициализация списков для хранения результатов
    class_ids = []
    confidences = []
    boxes = []
    class_counts = {class_name: 0 for class_name in classes}  # Словарь для подсчета объектов каждого класса

    # Анализ выходов нейронной сети
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_score = max(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применение немаксимального подавления
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Подсчет и отрисовка объектов
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]].tolist()  # Цвет для данного класса
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidences[i]:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Увеличение счетчика для класса
            class_counts[label] += 1

    # Вывод подсчета объектов в консоль
    for class_name, count in class_counts.items():
        if count > 0:  # Выводим только классы, где обнаружены объекты
            print(f"{class_name}: {count}")

    return image

# # Функция для детектирования объектов на видео
def detect_objects_on_video(video_path):
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
            # Детектирование объектов на кадре
            frame = detect_objects_on_image(frame)

            # Отображение кадра
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
    args = parser.parse_args()

    if args.image:
        # Детектирование объектов на изображении
        processed_image = detect_objects_on_image(args.image)
        if processed_image is not None:
            cv2.imshow("Detected Image", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif args.video:
        # Детектирование объектов на видео
        detect_objects_on_video(args.video)
    else:
        print("Error: Please provide an image or video file path using --image or --video.")