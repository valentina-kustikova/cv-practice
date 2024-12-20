import cv2
import numpy as np
import argparse

MODEL_CONFIG = "yolov3.cfg"
MODEL_WEIGHTS = "yolov3.weights"
CLASSES_FILE = "coco.names"

#Загрузка классов
with open(CLASSES_FILE, "r") as f:
    CLASSES = f.read().strip().split("\n")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#Загрузка модели
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_objects(image, confidence_threshold, nms_threshold):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    detections = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    results = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        results.append((class_ids[i], confidences[i], (x, y, w, h)))

    return results

def draw_detections(image, detections):
    class_counts = {cls: 0 for cls in CLASSES}

    for class_id, confidence, box in detections:
        x, y, w, h = box
        color = COLORS[class_id]
        label = f"{CLASSES[class_id]}"
        confidence_text = f"{confidence:.3f}"

        class_counts[CLASSES[class_id]] += 1

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(image, confidence_text, (x + 1, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, class_counts

def process_image(image_path, confidence_threshold, nms_threshold):
    image = cv2.imread(image_path)
    detections = detect_objects(image, confidence_threshold, nms_threshold)
    output_image, class_counts = draw_detections(image, detections)

    print("Обнаруженные объекты:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    cv2.imshow("Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, confidence_threshold, nms_threshold):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame, confidence_threshold, nms_threshold)
        output_frame, class_counts = draw_detections(frame, detections)

        print("Обнаруженные объекты:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count}")
        print("\n")

        cv2.imshow("Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Object Detection using OpenCV DNN.")
    parser.add_argument("input_type", choices=["image", "video"], help="Тип ввода: image или video.")
    parser.add_argument("input_path", help="Путь к изображению или видео.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Порог достоверности (по умолчанию 0.5).")
    parser.add_argument("--nms", type=float, default=0.4, help="Порог NMS (по умолчанию 0.4).")
    args = parser.parse_args()

    if args.input_type == "image":
        process_image(args.input_path, args.confidence, args.nms)
    elif args.input_type == "video":
        process_video(args.input_path, args.confidence, args.nms)

if __name__ == "__main__":
    main()
