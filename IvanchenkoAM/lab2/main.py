import cv2 as cv
import numpy as np
import argparse

class YOLODetector:
    def __init__(self, weights, config, names):
        self.net = cv.dnn.readNet(weights, config)
        with open(names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.output_layers = [self.net.getLayerNames()[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame, conf_threshold=0.5, nms_threshold=0.4, input_size=(416, 416)):
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 0.00392, input_size, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, classes, confidences, class_ids = [], [], [], []
        detected_objects = {}

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                    width, height = int(detection[2] * w), int(detection[3] * h)
                    x, y = int(center_x - width / 2), int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    classes.append(class_id)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, width, height = boxes[i]
                detected_objects[self.classes[classes[i]]] = detected_objects.get(self.classes[classes[i]], 0) + 1
                color = self.colors[class_ids[i]]
                label = f"{self.classes[class_ids[i]]}: {confidences[i]:.3f}"
                cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detected_objects

    def process_frame(self, frame, conf_threshold, nms_threshold, input_size):
        output_frame, detected_objects = self.detect_objects(frame, conf_threshold, nms_threshold, input_size)
        print('Objects in the frame:')
        for obj, count in detected_objects.items():
            print(f"{obj}: {count}")
        return output_frame
'''
    def process_batch(self, batch, conf_threshold, nms_threshold, input_size):
        output_batch, detected_objects_batch = self.detect_objects(batch, conf_threshold, nms_threshold, input_size)
        for obj, count in detected_objects.items():
            print(f"{obj}: {count}")
        return output_frame
'''
def display_frame(frame, window_name="Object Detection", wait_for_key=False):
   cv.imshow(window_name, frame)
   if wait_for_key:
       key = cv.waitKey(0)
   else:
       key = cv.waitKey(1)
   return key != ord("q")
def cleanup():
    cv.destroyAllWindows()


def process_input(input_path, yolo, conf_threshold, nms_threshold, input_size):
    if input_path.endswith((".mp4", ".avi")):
        vidcap = cv.VideoCapture(input_path)
        frame_width = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = vidcap.get(cv.CAP_PROP_FPS)
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter('output_video_path.mp4', fourcc, fps/4, (frame_width, frame_height))
        success,frame = vidcap.read()
        count = 0
        #frame_list = list()
        while success:
            output_frame = yolo.process_frame(frame, conf_threshold, nms_threshold, input_size)
            cv.imshow('detection', output_frame)
            if cv.waitKey(30) > 0:
                print("Stopped by user")
                break
            #frame_list.append(frame)
            #cv.imwrite("frames/frame%d.jpg" % count, output_frame)     # save frame as JPEG file      
            out.write(output_frame)
            success,frame = vidcap.read()
            count += 1 
        vidcap.release()
        out.release() 
        '''
            success,frame = vidcap.read()
            count = 0
            frame_list = list()
            while success:
                frame_list.append(frame)
            
            batch_size = (len(frame_list), input_size[0], input_size[1])
            output_frame = yolo.process_frame(frame_list, conf_threshold, nms_threshold, batch_size)
            
            for f in output_frame:
                #cv.imwrite("frames/frame%d.jpg" % count, output_frame)     # save frame as JPEG file      
                out.write(f)
                success,frame = vidcap.read()
                print('Read a new frame: ', success)
                count += 1    
            vidcap.release()
            out.release()     
        ''' 
    else:
        image = cv.imread(input_path)
        if image is None:
            print("Error: Could not load the image.")
            return
        output_image = yolo.process_frame(image, conf_threshold, nms_threshold, input_size)
        display_frame(output_image, wait_for_key=True)
    cleanup()



def main():
    parser = argparse.ArgumentParser(description="Object Detection with YOLO")
    parser.add_argument('-i', '--input', required=True, help="Path to the input image or video")
    parser.add_argument('-c', '--confidence_threshold', type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument('-n', '--nms_threshold', type=float, default=0.4, help="NMS threshold (default: 0.4)")
    parser.add_argument('-s', '--input_size', type=int, nargs=2, default=[416, 416], help="YOLO input size (default: 416 416)")
    parser.add_argument('-w', '--weights', required=True, help="Path to YOLO weights file")
    parser.add_argument('-cfg', '--config', required=True, help="Path to YOLO config file")
    parser.add_argument('-names', '--names', required=True, help="Path to class names file")

    args = parser.parse_args()
    yolo = YOLODetector(args.weights, args.config, args.names)
    process_input(args.input, yolo, args.confidence_threshold, args.nms_threshold, tuple(args.input_size))


if __name__ == '__main__':
    main()