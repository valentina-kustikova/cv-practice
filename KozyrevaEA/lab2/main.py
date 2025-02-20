import cv2 as cv
import numpy as np
import argparse
from typing import List, Tuple, Dict, Any


class ObjectDetector:
    """Class for performing object detection using SSD MobileNet model"""

    CLASSES: Tuple[str, ...] = ("background", 
                                "aeroplane", "bicycle", "bird", "boat", 
                                "bottle", "bus", "car", "cat", "chair", 
                                "cow", "diningtable", "dog", "horse", 
                                "motorbike", "person", "pottedplant", 
                                "sheep", "sofa", "train", "tvmonitor"
    )
    COLORS: np.ndarray = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    def __init__(self, 
                 weight_file: str, 
                 config_file: str) -> None:
        self.model = self.initialize_model(weight_file, config_file)


    def initialize_model(self, 
                         weight_file: str, 
                         config_file: str) -> cv.dnn_Net:
        """Load model from provided weight and configuration files"""

        net = cv.dnn.readNetFromCaffe(config_file, weight_file)
        return net


    def create_input_tensor(self, 
                            image: np.ndarray, 
                            size: Tuple[int, int] = (300, 300), 
                            scale: float = 0.007843, 
                            mean: Tuple[float, float, float] = (127.5, 127.5, 127.5)) -> np.ndarray:
        """Create blob from image for model input"""

        return cv.dnn.blobFromImage(image, scale, size, mean, swapRB=True)


    def fetch_predictions(self, 
                          input_blob: np.ndarray, 
                          h: int, 
                          w: int, 
                          confidence_threshold: float = 0.5) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """Get and parse predictions from model"""

        self.model.setInput(input_blob)
        detections = self.model.forward()

        results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
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


    def draw_bbox(self, image: np.ndarray, 
                  bbox: List[Tuple[int, float, Tuple[int, int, int, int]]]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Draw bounding boxes on image and count detected objects"""

        detected: Dict[str, int] = {}
        for class_id, confidence, (x1, y1, x2, y2) in bbox:
            label = self.CLASSES[class_id]
            color = self.COLORS[class_id]

            detected[label] = detected.get(label, 0) + 1

            text = f"{label}: {confidence:.3f}"
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv.putText(image, f"{confidence:.3f}", (x1, y1 - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image, detected


class ImageProcessor:
    def __init__(self, 
                 detector: ObjectDetector) -> None:
        self.detector = detector


    def load_image(self, 
                   image_path: str) -> np.ndarray:
        """Load image from path"""

        image = cv.imread(image_path)

        return image

    def resize_image(self, 
                     image: np.ndarray) -> np.ndarray:
        """Resize image"""

        return cv.resize(image, (image.shape[1] // 1, image.shape[0] // 1))


    def process_image(self, 
                      image: np.ndarray, 
                      conf_threshold: float) -> Tuple[np.ndarray, Dict[str, int]]:
        """Process image and return processed image with detected objects"""

        h, w = image.shape[:2]
        blob = self.detector.create_input_tensor(image)
        predictions = self.detector.fetch_predictions(blob, h, w, conf_threshold)

        if not predictions:
            print("Model did not detect any objects in image")
            return image, {}  

        return self.detector.draw_bbox(image, predictions)


    def display_results(self, 
                        image: np.ndarray, 
                        detected_objects: Dict[str, int]) -> None:
        """Display results and detected objects"""

        for label, count in detected_objects.items():
            print(f"Detected {count} object(s) of class: {label}")

        cv.imshow("Object Detection", image)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def handle_image(self, 
                     image_path: str, 
                     conf_threshold: float = 0.5) -> None:
        """Handle process of loading, processing, and displaying image"""

        image = self.load_image(image_path)
        resized_image = self.resize_image(image)
        processed_image, detected_objects = self.process_image(resized_image, conf_threshold)
        self.display_results(processed_image, detected_objects)

class VideoProcessor:
    def __init__(self, detector: ObjectDetector) -> None:
        self.detector = detector


    def handle_video(self, 
                     video_path: str, 
                     conf_threshold: float = 0.5, 
                     target_width: int = 640, 
                     target_height: int = 480) -> None:
        """Process video frame by frame and display object detection results"""

        cap = cv.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv.resize(frame, (target_width, target_height))

            processed_frame = self.handle_image_frame(resized_frame, conf_threshold)
            cv.imshow("Object Detection", processed_frame)

            if cv.waitKey(1) & 0xFF == ord("k"):
                break

        cap.release()
        cv.destroyAllWindows()


    def handle_image_frame(self, 
                           frame: np.ndarray, 
                           conf_threshold: float = 0.5) -> np.ndarray:
        """Handle processing of single video frame"""

        h, w = frame.shape[:2]
        blob = self.detector.create_input_tensor(frame)
        predictions = self.detector.fetch_predictions(blob, h, w, conf_threshold)
        processed_frame, detected_objects = self.detector.draw_bbox(frame, predictions)

        for label, count in detected_objects.items():
            print(f"Detected {count} object(s) of class: {label}")

        return processed_frame


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for input and confidence threshold"""

    parser = argparse.ArgumentParser(description="SSD MobileNet Object Detection")
    parser.add_argument("-i", "--input", required=True, help="Path to image or video")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold for object detection")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    detector = ObjectDetector("models\mobilenet_iter_73000.caffemodel", "models\deploy.prototxt")

    if args.input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        print(f"Processing image: {args.input}")
        image_processor = ImageProcessor(detector)
        image_processor.handle_image(args.input, args.confidence)

    elif args.input.lower().endswith((".mp4", ".avi", ".mov")):
        print(f"Processing video: {args.input}")
        video_processor = VideoProcessor(detector)
        video_processor.handle_video(args.input, args.confidence)


if __name__ == "__main__":
    main()
