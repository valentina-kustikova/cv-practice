import cv2
import numpy as np
import argparse
import os

class YOLOv4Detector:
    def __init__(self, weights_path, config_path, classes_path, conf_threshold=0.5, nms_threshold=0.4):
        self.net, self.layersOutput = self.loadModel(weights_path, config_path)
        self.classes = self.loadClasses(classes_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def loadClasses(self, classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def loadModel(self, weights_path, config_path):
        net = cv2.dnn.readNet(weights_path, config_path)
        layerNames = net.getLayerNames()
        layersOutput = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, layersOutput

    def preprocessImage(self, image):
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        return height, width, channels

    def objectsDetection(self, height, width, channels, image):
        output = self.net.forward(self.layersOutput)
        output = np.concatenate(output, axis=0)
        observations = output[:, 5:]
        classIds = np.argmax(observations, axis=1)
        confidences = np.max(observations, axis=1)
        valid = confidences > self.conf_threshold
        boxes = output[:, :4]
        centerX = (boxes[:, 0] * width).astype(int)
        centerY = (boxes[:, 1] * height).astype(int)
        widthRectangle = (boxes[:, 2] * width).astype(int)
        heightRectangle = (boxes[:, 3] * height).astype(int)
        leftX = centerX - widthRectangle // 2
        leftY = centerY - heightRectangle // 2
        valid_leftX = leftX[valid]
        valid_leftY = leftY[valid]
        valid_widthRectangle = widthRectangle[valid]
        valid_heightRectangle = heightRectangle[valid]
        valid_confidences = confidences[valid]
        valid_classIds = classIds[valid]
        rectangles = np.column_stack((valid_leftX, valid_leftY, valid_widthRectangle, valid_heightRectangle))
        indexes = cv2.dnn.NMSBoxes(rectangles.tolist(), valid_confidences.tolist(), self.conf_threshold, self.nms_threshold)

        detected_objects = {}
        for i in indexes.flatten():
            class_id = valid_classIds[i]
            detected_objects[class_id] = detected_objects.get(class_id, 0) + 1

        return rectangles, valid_confidences, valid_classIds, indexes, detected_objects

    def showResults(self, image, rectangles, confidences, class_ids, indexes, detected_objects):
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        for i in indexes.flatten():
            leftX, leftY, widthRectangle, heightRectangle = rectangles[i]
            label = str(self.classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (leftX, leftY), (leftX + widthRectangle, leftY + heightRectangle), color, 2)
            cv2.putText(image, f'{label} {confidences[i]:.3f}', (leftX, leftY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        y_offset = 30
        for class_id, count in detected_objects.items():
            label = f"{self.classes[class_id]}: {count}"
            cv2.putText(image, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 30

        return image

    def readImage(self, image_path):
        if image_path is None:
            raise ValueError('Empty path to the image')
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError('Unable to load image')
        return image

    def writeImage(self, output_image, result_image):
        success = cv2.imwrite(output_image, result_image)
        if not success:
            print(f"Error saving image to {output_image}")

    def showImage(self, image):
        cv2.imshow('Result', image)
        cv2.waitKey(0)

    def processVideo(self, video_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError('Unable to open video file')

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width, channels = self.preprocessImage(frame)
            rectangles, confidences, classIDS, indexes, detected_objects = self.objectsDetection(height, width, channels, frame)
            result_frame = self.showResults(frame, rectangles, confidences, classIDS, indexes, detected_objects)

            cv2.imshow('Result', result_frame)
            output_path = os.path.join(output_dir, f'frame_{frame_number:04d}.jpg')
            self.writeImage(output_path, result_frame)
            frame_number += 1

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                continue
            elif key == ord('e'):
                break

        cap.release()
        cv2.destroyAllWindows()

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\' or \'video\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--input',
                        help='Path to an image or video',
                        type=str,
                        dest='input_path')
    parser.add_argument('-o', '--output',
                        help='Output image or video name',
                        type=str,
                        dest='output_path',
                        default='output.jpg')
    parser.add_argument('-d', '--output_dir',
                        help='Output directory for saving frames',
                        type=str,
                        dest='output_dir',
                        default='output_frames')
    parser.add_argument('-c', '--config',
                        help='Path to the config file',
                        type=str,
                        dest='config_path',
                        default='yolov4.cfg')
    parser.add_argument('-w', '--weights',
                        help='Path to the weights file',
                        type=str,
                        dest='weights_path',
                        default='yolov4.weights')
    parser.add_argument('-cl', '--classes',
                        help='Path to the classes file',
                        type=str,
                        dest='classes_path',
                        default='coco.names')
    parser.add_argument('-ct', '--conf_threshold',
                        help='Confidence threshold',
                        type=float,
                        dest='conf_threshold',
                        default=0.5)
    parser.add_argument('-nt', '--nms_threshold',
                        help='NMS threshold',
                        type=float,
                        dest='nms_threshold',
                        default=0.4)

    args = parser.parse_args()
    return args

def main():
    args = cli_argument_parser()
    detector = YOLOv4Detector(args.weights_path, args.config_path, args.classes_path, args.conf_threshold, args.nms_threshold)

    if args.mode == 'image':
        image = detector.readImage(args.input_path)
        height, width, channels = detector.preprocessImage(image)
        rectangles, confidences, classIDS, indexes, detected_objects = detector.objectsDetection(height, width, channels, image)
        resultImage = detector.showResults(image, rectangles, confidences, classIDS, indexes, detected_objects)
        detector.writeImage(args.output_path, resultImage)
        detector.showImage(resultImage)
    elif args.mode == 'video':
        detector.processVideo(args.input_path, args.output_dir)
    else:
        raise ValueError('Unsupported \'mode\' value')

if __name__ == '__main__':
    main()
