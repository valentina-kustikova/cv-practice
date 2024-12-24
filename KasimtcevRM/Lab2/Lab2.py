import cv2 as cv
import numpy as np
import argparse
import os
class detecter:
    def __init__(par, _w, _c, _l, _c_t, _n_t):
        par.model, par.output_names = par.Ld_Model(_w, _c)
        par.l = par.Ld_label(_l)
        par.c_t = _c_t
        par.n_t = _n_t

    def Ld_label(par, _l):
        with open(_l, 'r') as _file:
            l = [line.strip() for line in _file.readlines()]
        return l

    def Ld_Model(par, _w, _c):
        model = cv.dnn.readNet(_w, _c)
        layer_names = model.getLayerNames()
        output_names = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
        return model, output_names

    def Process_Image(par, _image):
        height, width, channels = _image.shape
        blob = cv.dnn.blobFromImage(_image, 1/255, (608, 608), (0, 0, 0), True, crop=False)
        par.model.setInput(blob)
        return height, width, channels

    def Process_Output(par, _output, _width, _height):
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

    def Detect_Image(par, _image):
        height, width, channels = par.Process_Image(_image)

        output = par.model.forward(par.output_names)
        class_ids, confidences, rectangles = par.Process_Output(output, width, height)

        indexes = cv.dnn.NMSBoxes(rectangles.tolist(), confidences.tolist(), par.c_t, par.n_t)

        detections = {}
        for i in indexes:
            class_id = class_ids[i]
            detections[class_id] = (1) if (class_id not in detections.keys()) else (detections[class_id] + 1)

        return rectangles, confidences, class_ids, indexes, detections

class Visualisator:
    def __init__(par, _l):
        par.l = _l

    def Im_rezuilt(par, _image, _rectangles, _confidences, _class_ids, _indexes, _detections, _colors):
        for i in _indexes:
            left_x, left_y, rect_width, rect_height = _rectangles[i]
            label = str(par.l[_class_ids[i]])
            cv.rectangle(_image, (left_x, left_y), (left_x + rect_width, left_y + rect_height), _colors[_class_ids[i]], 2)
            cv.putText(_image, f'{label} {_confidences[i]:.3f}', (left_x - 0, left_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, _colors[_class_ids[i]], 2)

        label_offset = [0, 20]
        for class_id, count in _detections.items():
            label = f"{par.l[class_id]}: {count}"
            cv.putText(_image, label, label_offset, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            label_offset[0] += 0
            label_offset[1] += 20

        return _image
    
    def Show_Image(par, _image, _window_name = 'Detection result', _wait = True):
        cv.imshow(_window_name, _image)
        if _wait:
            cv.waitKey(0)


class FileManage:
    def Read_Image(par, _image):
        img = cv.imread(_image)
        if not img.any():
            raise ValueError("Unable to read image")
        else:
            return img
        
    def Write_Image(par, _image, _output_path):
        return cv.imwrite(_output_path, _image)
    
    def Process_Video(par, _detecter, _Visualisator, _colors, _video_path, _output_path, _window_name = 'Detection result', _fps = 30, _frame_size = (608, 608)):
        capture = cv.VideoCapture(_video_path)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            rectangles, confidences, class_ids, indexes, detections = _detecter.Detect_Image(frame)
            result_frame = _Visualisator.Im_rezuilt(frame, rectangles, confidences, class_ids, indexes, detections, _colors)
            _Visualisator.Show_Image(result_frame, _window_name, False)
            cv.waitKey(max(int(1/_fps), 1))
            a=cv.waitKey(1)
            if a == ord("q"):
                break
        capture.release()
        cv.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Object Detection with YOLO")
    parser.add_argument('-i', '--input',help='Path to an image or video.',required=True,type=str,dest='input')
    parser.add_argument('-o', '--output',help='Output path.',type=str,dest='output')
    parser.add_argument('-m', '--mode',help='Input mode.',required=True,choices=['Image','Video'],type=str,dest='mode')
    parser.add_argument('-c', '--config',help='Path to the c file.',required=True,type=str,dest='c')
    parser.add_argument('-w', '--weights',help='Path to the w file.',required=True,type=str,dest='w')
    parser.add_argument('-l', '--labels',help='Path to the l file.',required=True,type=str, dest='l')
    parser.add_argument('-ct', '--conf_threshold',help='Confidence threshold.',type=float,dest='c_t',default=0.5)
    parser.add_argument('-nt', '--nms_threshold',help='Non-maximum suppression threshold.',type=float,dest='n_t',default=0.5)
    args = parser.parse_args()
    detector = detecter(args.w, args.c, args.l, args.c_t, args.n_t)
    visualisator = Visualisator(detector.l)
    file_manage = FileManage()
    colors = np.random.uniform(0, 255, size=(len(detector.l), 3))

    if args.mode == 'Image':
        image = file_manage.Read_Image(args.input)
        rectangles, confidences, class_ids, indexes, detections = detector.Detect_Image(image)
        result = visualisator.Im_rezuilt(image, rectangles, confidences, class_ids, indexes, detections, colors)
        visualisator.Show_Image(result)
        if args.output != None:
            file_manage.Write_Image(result, args.output)
    else:
        file_manage.Process_Video(detector, visualisator, colors, args.input, args.output)

if __name__ == '__main__':
    main()
