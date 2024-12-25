import cv2 as cv
import numpy as np
import argparse

def initialize_model(weight_file, config_file):
    net = cv.dnn.readNetFromCaffe(config_file, weight_file)
    return net

def create_input_tensor(image, size=(300, 300), scale=0.007843, mean=(127.5, 127.5, 127.5)):
    return cv.dnn.blobFromImage(image, scale, size, mean, swapRB=True)

def fetch_predictions(model, input_blob, h, w, confidence_threshold=0.5):
    model.setInput(input_blob)
    detections = model.forward()

    results = []
    for detection in detections[0, 0]:
        confidence = detection[2]
        if confidence > confidence_threshold:
            class_id = int(detection[1])
            if class_id < len(CLASSES):
                x1, y1, x2, y2 = (detection[3:7] * [w, h, w, h]).astype("int")
                results.append((class_id, confidence, (x1, y1, x2, y2)))
    return results

def draw_predictions(image, predictions):
    detected = {}

    for class_id, confidence, (x1, y1, x2, y2) in predictions:
        label = CLASSES[class_id]
        color = COLORS[class_id]

        detected[label] = detected.get(label, 0) + 1

        text_label = f"{label}: {confidence:.3f}"
        text_conf = f"{confidence:.3f}"

        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(image, text_label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.putText(image, text_conf, (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, detected


def handle_image(image_path, model, conf_threshold=0.5):
    image = cv.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    h, w = image.shape[:2]

    blob = create_input_tensor(image)
    predictions = fetch_predictions(model, blob, h, w, conf_threshold)

    processed_image, detected_objects = draw_predictions(image, predictions)

    for label, count in detected_objects.items():
        print(f"Обнаружено {count} объекта(ов) класса: {label}")

    cv.imshow("Object Detection", processed_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def handle_video(video_path, model, conf_threshold=0.5):
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = handle_image_frame(frame, model, conf_threshold)

        cv.imshow("Object Detection", processed_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


#Обработка одного кадра видео
def handle_image_frame(frame, model, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = create_input_tensor(frame)
    predictions = fetch_predictions(model, blob, h, w, conf_threshold)
    processed_frame, detected_objects = draw_predictions(frame, predictions)

    for label, count in detected_objects.items():
        print(f"Обнаружено {count} объекта(ов) класса: {label}")

    return processed_frame

#Основная функция для запуска из командной строки
def main():
    parser = argparse.ArgumentParser(description="SSD MobileNet Object Detection")
    parser.add_argument('-i', '--input', required=True, help="Путь к изображению или видео")
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help="Порог уверенности для детектирования объектов")
    args = parser.parse_args()

    #Инициализмруем модель и классы
    model = initialize_model('mobilenet_iter_73000.caffemodel', 'deploy.prototxt')
    global CLASSES, COLORS
    CLASSES = ('background', 
               'aeroplane', 'bicycle', 'bird', 'boat', 
               'bottle', 'bus', 'car', 'cat', 'chair', 
               'cow', 'diningtable', 'dog', 'horse', 
               'motorbike', 'person', 'pottedplant', 
               'sheep', 'sofa', 'train', 'tvmonitor')
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    #Обработка изображения или видео
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"Обработка изображения: {args.input}")
        handle_image(args.input, model, args.confidence)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Обработка видео: {args.input}")
        handle_video(args.input, model, args.confidence)
    else:
        print("Неподдерживаемый формат файла")

if __name__ == "__main__":
    main()