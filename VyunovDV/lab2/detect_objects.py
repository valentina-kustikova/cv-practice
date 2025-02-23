import cv2
import numpy as np
from typing import List, Tuple
import argparse
import mimetypes

parser = argparse.ArgumentParser('Detect')
parser.add_argument('--proto', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--media_path', type=str)
args = parser.parse_args()

# Загрузка предварительно обученной модели
net = cv2.dnn.readNetFromCaffe(args.proto, args.model_path)

def is_video_or_photo_mime(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        if mime_type.startswith('video'):
            return "video"
        elif mime_type.startswith('image'):
            return "image"
    return "Неизвестный тип"


def detect_objects(image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    blob = cv2.dnn.blobFromImage(image, 1 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 7:
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                results.append((startX, startY, endX, endY, confidence))
    return results

def show_detected_image(image_path: str) -> None:
    if is_video_or_photo_mime(image_path) == 'image':
        image = cv2.imread(image_path)
        results = detect_objects(image)
        for (startX, startY, endX, endY, confidence) in results:
            label = f"Car: {confidence:.3f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite('inference.jpg', image)
        return 0
    elif is_video_or_photo_mime(image_path) == 'video':
        cap = cv2.VideoCapture(image_path)
        while True:
            _, image = cap.read()
            if _:
                results = detect_objects(image)
                for (startX, startY, endX, endY, confidence) in results:
                    label = f"Car: {confidence:.3f}"
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if cv2.waitKey(1) == ord('q'):
                    break
                cv2.imshow('window', image)
            else:
                break


image = cv2.imread(args.media_path)
show_detected_image(args.media_path)
