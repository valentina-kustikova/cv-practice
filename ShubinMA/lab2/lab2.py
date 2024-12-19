import cv2 as cv
import numpy as np
import argparse
import os


BB_TEXT_OFFSET_X = 0
BB_TEXT_OFFSET_Y = 10
LABEL_TEXT_OFFSET_X = 0
LABEL_TEXT_OFFSET_Y = 20


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Input mode.',
                        required=True,
                        choices=['Image', 'Video', 'VideoStream'],
                        type=str,
                        dest='mode')
    parser.add_argument('-i', '--input',
                        help='Path to an image or video.',
                        required=True,
                        type=str,
                        dest='input')
    parser.add_argument('-o', '--output',
                        help='Output path.',
                        type=str,
                        dest='output')
    parser.add_argument('-c', '--config',
                        help='Path to the config file.',
                        required=True,
                        type=str,
                        dest='config_path')
    parser.add_argument('-w', '--weights',
                        help='Path to the weights file.',
                        required=True,
                        type=str,
                        dest='weights_path')
    parser.add_argument('-l', '--labels',
                        help='Path to the labels file.',
                        required=True,
                        type=str,
                        dest='labels_path')
    parser.add_argument('-ct', '--conf_threshold',
                        help='Confidence threshold.',
                        type=float,
                        dest='conf_threshold',
                        default=0.5)
    parser.add_argument('-nt', '--nms_threshold',
                        help='Non-maximum suppression threshold.',
                        type=float,
                        dest='nms_threshold',
                        default=0.5)

    args = parser.parse_args()
    return args


class Detector:
    def __init__(self, _weights_path, _config_path, _labels_path, _conf_threshold, _nms_threshold):
        self.model, self.output_names = self.LoadModel(_weights_path, _config_path)
        self.labels = self.LoadLabels(_labels_path)
        self.conf_threshold = _conf_threshold
        self.nms_threshold = _nms_threshold

    def LoadLabels(self, _labels_path):
        with open(_labels_path, 'r') as _file:
            labels = [line.strip() for line in _file.readlines()]
        return labels

    def LoadModel(self, _weights_path, _config_path):
        model = cv.dnn.readNet(_weights_path, _config_path)
        layer_names = model.getLayerNames()
        output_names = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
        return model, output_names

    def PreprocessImage(self, _image):
        raise NotImplementedError("Image preprocessing function not implemented in class {(self.__class__.__name__)}.")

    def ProcessOutput(self, _output, _width, _height):
        raise NotImplementedError("Model output processing function not implemented in class {(self.__class__.__name__)}.")

    def DetectOnImage(self, _image):
        height, width, channels = self.PreprocessImage(_image)

        output = self.model.forward(self.output_names)
        class_ids, confidences, rectangles = self.ProcessOutput(output, width, height)

        indexes = cv.dnn.NMSBoxes(rectangles.tolist(), confidences.tolist(), self.conf_threshold, self.nms_threshold)

        detections = {}
        for i in indexes:
            class_id = class_ids[i]
            detections[class_id] = (1) if (class_id not in detections.keys()) else (detections[class_id] + 1)

        return rectangles, confidences, class_ids, indexes, detections

    def DetectOnVideo(self, _frames):
        results = []
        for i in range(len(_frames)):
            rectangles, confidences, class_ids, indexes, detections = self.DetectOnImage(_frames[i])
            results.append((rectangles, confidences, class_ids, indexes, detections))
            print("Frame", i+1, "of", len(_frames), "processed (detection).")

        return results

class YOLOv4tfDetector(Detector):
    def __init__(self, _weights_path, _config_path, _labels_path, _conf_threshold, _nms_threshold): 
        super().__init__(_weights_path, _config_path, _labels_path, _conf_threshold, _nms_threshold)

    def PreprocessImage(self, _image):
        height, width, channels = _image.shape
        blob = cv.dnn.blobFromImage(_image, 1/255, (608, 608), (0, 0, 0), True, crop=False)
        self.model.setInput(blob)
        return height, width, channels

    def ProcessOutput(self, _output, _width, _height):
        output = np.concatenate(_output, axis=0)
        class_ids = np.argmax(output[:, 5:], axis=1)
        confidences = np.max(output[:, 5:], axis=1)

        boxes = output[:, :4]
        center_x = (boxes[:, 0] * _width).astype(int)
        center_y = (boxes[:, 1] * _height).astype(int)
        rect_width = (boxes[:, 2] * _width).astype(int)
        rect_height = (boxes[:, 3] * _height).astype(int)
        left_x = center_x - rect_width // 2
        left_y = center_y - rect_height // 2
        rectangles = np.column_stack((left_x, left_y, rect_width, rect_height))

        return class_ids, confidences, rectangles


class Visualizer:
    def __init__(self, _labels):
        self.labels = _labels

    def ApplyImageResults(self, _image, _rectangles, _confidences, _class_ids, _indexes, _detections, _colors):
        for i in _indexes:
            left_x, left_y, rect_width, rect_height = _rectangles[i]
            label = str(self.labels[_class_ids[i]])
            cv.rectangle(_image, (left_x, left_y), (left_x + rect_width, left_y + rect_height), _colors[_class_ids[i]], 2)
            cv.putText(_image, f'{label} {_confidences[i]:.3f}', (left_x - BB_TEXT_OFFSET_X, left_y - BB_TEXT_OFFSET_Y), cv.FONT_HERSHEY_SIMPLEX, 0.5, _colors[_class_ids[i]], 2)

        label_offset = [LABEL_TEXT_OFFSET_X, LABEL_TEXT_OFFSET_Y]
        for class_id, count in _detections.items():
            label = f"{self.labels[class_id]}: {count}"
            cv.putText(_image, label, label_offset, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            label_offset[0] += LABEL_TEXT_OFFSET_X
            label_offset[1] += LABEL_TEXT_OFFSET_Y

        return _image

    def ApplyVideoResults(self, _frames, _processed_frames, _colors):
        frames = []
        for i in range(len(_frames)):
            result_frame = self.ApplyImageResults(_frames[i], _processed_frames[i][0], _processed_frames[i][1], _processed_frames[i][2], _processed_frames[i][3], _processed_frames[i][4], _colors)
            frames.append(result_frame)
            print("Frame", i+1, "of", len(_frames), "processed (results application).")

        return frames

    def ShowImage(self, _image, _window_name = 'Detection result', _wait = True):
        cv.imshow(_window_name, _image)
        if _wait:
            cv.waitKey(0)

    def ShowVideo(self, _frames, _window_name = 'Detection result', _fps=30, _wait = True):
        for i in range(len(_frames)):
            self.ShowImage(_frames[i], _window_name, False)
            if (_wait):
                cv.waitKey(max(int(1/_fps), 1))


class IOManager:
    def ReadImage(self, _image_path):
        img = cv.imread(_image_path)
        if not img.any():
            raise ValueError("Unable to read image")
        else:
            return img

    def WriteImage(self, _image, _output_path):
        return cv.imwrite(_output_path, _image)

    def ReadVideo(self, _video_path):
        capture = cv.VideoCapture(_video_path)
        if not capture.isOpened():
            raise ValueError('Unable to read video file')

        frames = []
        while True:
            ret, frame = capture.read()

            if not ret:
                break

            frames.append(frame)

        capture.release()

        return frames

    def WriteVideo(self, _frames, _output_path, _fps=30):
        height, width, channels = _frames[0].shape

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(_output_path, fourcc, _fps, (width, height))

        for frame in _frames:
            out.write(frame)

        out.release()

    def ProcessVideoStream(self, _detector, _visualizer, _colors, _video_path, _output_path, _window_name = 'Detection result', _fps = 30, _frame_size = (608, 608)):
        capture = cv.VideoCapture(_video_path)
        if not capture.isOpened():
            raise ValueError('Unable to read video file')

        if _output_path != None:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            out = cv.VideoWriter(_output_path, fourcc, _fps, _frame_size)

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            rectangles, confidences, class_ids, indexes, detections = _detector.DetectOnImage(frame)
            result_frame = _visualizer.ApplyImageResults(frame, rectangles, confidences, class_ids, indexes, detections, _colors)

            _visualizer.ShowImage(result_frame, _window_name, False)

            cv.waitKey(max(int(1/_fps), 1))

            if _output_path != None:
                out.write(frame)

        if _output_path != None:
            out.release()
        capture.release()
        cv.destroyAllWindows()


def main():
    args = cli_argument_parser()
    detector = YOLOv4tfDetector(args.weights_path, args.config_path, args.labels_path, args.conf_threshold, args.nms_threshold)
    visualizer = Visualizer(detector.labels)
    io_manager = IOManager()
    colors = np.random.uniform(0, 255, size=(len(detector.labels), 3))

    if args.mode == 'Image':
        image = io_manager.ReadImage(args.input)
        rectangles, confidences, class_ids, indexes, detections = detector.DetectOnImage(image)
        result = visualizer.ApplyImageResults(image, rectangles, confidences, class_ids, indexes, detections, colors)
        visualizer.ShowImage(result)
        if args.output != None:
            io_manager.WriteImage(result, args.output)
    elif args.mode == 'Video':
        frames = io_manager.ReadVideo(args.input)
        processed_frames = detector.DetectOnVideo(frames)
        results = visualizer.ApplyVideoResults(frames, processed_frames, colors)
        visualizer.ShowVideo(results, _wait = True)
        if args.output != None:
            io_manager.WriteVideo(results, args.output)
    else:
        io_manager.ProcessVideoStream(detector, visualizer, colors, args.input, args.output)

if __name__ == '__main__':
    main()