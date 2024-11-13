import cv2 as cv
import numpy as np
import argparse  # Импортируем argparse для парсинга командной строки

# Загрузка модели YOLO
net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Загрузим имена классов
with open('coco.names', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Получение имен выходных слоев модели
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Функция для детектирования объектов
def detect_objects(image, net, conf_threshold=0.5, nms_threshold=0.4, input_size=(416, 416)):
    (h, w) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 0.00392, input_size, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    detected_objects = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                if CLASSES[class_id] in detected_objects:
                    detected_objects[CLASSES[class_id]] += 1
                else:
                    detected_objects[CLASSES[class_id]] = 1

    # Применение NMS с порогом
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices.flatten():
        x, y, width, height = boxes[i]
        label = f"{CLASSES[class_ids[i]]}: {confidences[i]:.3f}"
        color = COLORS[class_ids[i]]
        
        cv.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, detected_objects

# Функция для обработки видео или изображения
def process_input(input_path, conf_threshold, nms_threshold, input_size):
    if input_path.endswith(".mp4") or input_path.endswith(".avi"):
        cap = cv.VideoCapture(input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            output_frame, detected_objects = detect_objects(frame, net, conf_threshold, nms_threshold, input_size)
            
            for obj, count in detected_objects.items():
                print(f"{obj}: {count}")
            
            cv.imshow("Object Detection", output_frame)
            
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv.destroyAllWindows()
    else:
        image = cv.imread(input_path)
        if image is None:
            print("Не удалось загрузить изображение")
            return
        
        output_image, detected_objects = detect_objects(image, net, conf_threshold, nms_threshold, input_size)
        
        for obj, count in detected_objects.items():
            print(f"{obj}: {count}")
        
        cv.imshow("Object Detection", output_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

# Основная функция для парсинга командной строки
def main():
    parser = argparse.ArgumentParser(description="Object Detection with YOLO")
    parser.add_argument('-i', '--input', required=True, help="Path to the input image or video")
    parser.add_argument('-c', '--confidence_threshold', type=float, default=0.5, help="Confidence threshold for object detection (default: 0.5)")
    parser.add_argument('-n', '--nms_threshold', type=float, default=0.4, help="NMS threshold for suppressing overlapping boxes (default: 0.4)")
    parser.add_argument('-s', '--input_size', type=int, nargs=2, default=[416, 416], help="Input size for the YOLO model (width height, default: 416 416)")


    args = parser.parse_args()
    input_path = args.input
    conf_threshold = args.confidence_threshold
    nms_threshold = args.nms_threshold
    input_size = tuple(args.input_size)
    
    process_input(input_path, conf_threshold, nms_threshold, input_size)

# Запуск программы
if __name__ == '__main__':
    main()
