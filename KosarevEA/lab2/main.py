import cv2 as cv
import numpy as np
import argparse


class ObjectDetector:
    """
    Класс для выполнения обнаружения объектов с использованием модели SSD MobileNet.
    """
    CLASSES = ('background', 
               'aeroplane', 'bicycle', 'bird', 'boat', 
               'bottle', 'bus', 'car', 'cat', 'chair', 
               'cow', 'diningtable', 'dog', 'horse', 
               'motorbike', 'person', 'pottedplant', 
               'sheep', 'sofa', 'train', 'tvmonitor')
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    def __init__(self, weight_file, config_file):
        self.model = self.initialize_model(weight_file, config_file)

    def initialize_model(self, weight_file, config_file):
        net = cv.dnn.readNetFromCaffe(config_file, weight_file)
        return net

    def create_input_tensor(self, image, size=(300, 300), scale=0.007843, mean=(127.5, 127.5, 127.5)):
        return cv.dnn.blobFromImage(image, scale, size, mean, swapRB=True)

    def fetch_predictions(self, input_blob, h, w, confidence_threshold=0.5):
        """
        Получение и разбор предсказаний модели.
        """
        self.model.setInput(input_blob)
        detections = self.model.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence <= confidence_threshold:
                continue 
            
            class_id = int(detections[0, 0, i, 1])

            if class_id >= len(self.CLASSES):
                continue 
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            results.append((class_id, confidence, (x1, y1, x2, y2)))
        
        return results

    def draw_bbox(self, image, bbox):
        detected = {}
        for class_id, confidence, (x1, y1, x2, y2) in bbox:
            label = self.CLASSES[class_id]
            color = self.COLORS[class_id]

            if label in detected:
                detected[label] += 1
            else:
                detected[label] = 1

            text = f"{label}: {confidence:.3f}"
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv.putText(image, f"{confidence:.3f}", (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image, detected


class ImageProcessor:
    def __init__(self, detector):
        self.detector = detector

    def load_image(self, image_path):
        image = cv.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None
        return image

    def resize_image(self, image):
        return cv.resize(image, (image.shape[1] // 1, image.shape[0] // 1))

    def process_image(self, image, conf_threshold):
        h, w = image.shape[:2]
        blob = self.detector.create_input_tensor(image)
        predictions = self.detector.fetch_predictions(blob, h, w, conf_threshold)

        if not predictions:
            print("Модель не обнаружила объектов на изображении.")
            return image, {}  

        return self.detector.draw_bbox(image, predictions)

    def display_results(self, image, detected_objects):
        for label, count in detected_objects.items():
            print(f"Обнаружено {count} объекта(ов) класса: {label}")

        cv.imshow("Object Detection", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def handle_image(self, image_path, conf_threshold=0.5):
        image = self.load_image(image_path)
        if image is None:
            return 

        resized_image = self.resize_image(image)

        processed_image, detected_objects = self.process_image(resized_image, conf_threshold)

        self.display_results(processed_image, detected_objects)


class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector

    def handle_video(self, video_path, conf_threshold=0.5, target_width=640, target_height=480):
        cap = cv.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv.resize(frame, (target_width, target_height))

            processed_frame = self.handle_image_frame(resized_frame, conf_threshold)
            cv.imshow("Object Detection", processed_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def handle_image_frame(self, frame, conf_threshold=0.5):
        h, w = frame.shape[:2]
        blob = self.detector.create_input_tensor(frame)
        predictions = self.detector.fetch_predictions(blob, h, w, conf_threshold)
        processed_frame, detected_objects = self.detector.draw_bbox(frame, predictions)

        for label, count in detected_objects.items():
            print(f"Обнаружено {count} объекта(ов) класса: {label}")

        return processed_frame


def parse_arguments():
    parser = argparse.ArgumentParser(description="SSD MobileNet Object Detection")
    parser.add_argument('-i', '--input', required=True, help="Путь к изображению или видео")
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help="Порог уверенности для детектирования объектов")
    return parser.parse_args()


def main():
    args = parse_arguments()
    detector = ObjectDetector('mobilenet_iter_73000.caffemodel', 'deploy.prototxt')

    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"Обработка изображения: {args.input}")
        image_processor = ImageProcessor(detector)
        image_processor.handle_image(args.input, args.confidence)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Обработка видео: {args.input}")
        video_processor = VideoProcessor(detector)
        video_processor.handle_video(args.input, args.confidence)
    else:
        print("Неподдерживаемый формат файла")


if __name__ == "__main__":
    main()