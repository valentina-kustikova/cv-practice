import cv2 as cv
import numpy as np
import argparse
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\', \'video\')',
                        type=str,
                        dest='mode',
                        default='image')
    
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    
    parser.add_argument('-v', '--video',
                        help='Path to video',
                        type=str,
                        dest='video_path')   
    
    parser.add_argument('-cf', '--cfg',
                        help='Path to an cfg file',
                        type=str,
                        dest='model_cfg')
    
    parser.add_argument('-w', '--weights',
                        help='Path to weights file',
                        type=str,
                        dest='model_weights')
    
    parser.add_argument('-cl', '--classes',
                        help='Path to classes file',
                        type=str,
                        dest='classes_file')
    
    parser.add_argument('-t', '--conf_threshold',
                        help='confidence threshold',
                        type=float,
                        dest='conf_threshold')    

    args = parser.parse_args()
    return args

def load_image(path, conf):

    img0 = cv.imread(path)
    
    img = img0.copy()
    
    outputs = process_image(img)
    
    post_process(img, outputs, conf)
    
    display_image(img)


def process_image(img):

    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(ln)
    outputs = np.vstack(outputs)  # объединяем все выходы в один массив

    return outputs   
    
def draw(img, indices, boxes, confidences, classIDs):
    if len(indices) > 0:
        obj_count = {}
        
        for i in indices.flatten():
            
            x, y, w, h = boxes[i]
            
            color = [int(c) for c in colors[classIDs[i]]]
            
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            text = f"{classes[classIDs[i]]}: {confidences[i]:.3f}"
            cv.putText(img, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            object_name = classes[classIDs[i]]
            
            obj_count[object_name] = obj_count.get(object_name, 0) + 1

        for key, value in obj_count.items():
            print(key, value)
            
            
def post_process(img, outputs, conf_threshold):

    H, W = img.shape[:2]
    boxes, confidences, classIDs = [], [], []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        
        if confidence > conf_threshold:
            
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            
            p0 = int(x - w // 2), int(y - h // 2)
            p1 = int(x + w // 2), int(y + h // 2)
            
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, conf_threshold - 0.1)
    
    draw(img, indices, boxes, confidences, classIDs) 

def display_image(img):

    cv.imshow('window', img)
    
    cv.waitKey(0)
    
    cv.destroyAllWindows()


def load_model(path_cfg, path_weights, path_classes):
    
    global net, ln, classes, colors
    
    classes = open(path_classes).read().strip().split('\n')
    
    np.random.seed(42)
    
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    net = cv.dnn.readNetFromDarknet(path_cfg, path_weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

    ln = net.getLayerNames()
    
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def process_video(video_path, conf_threshold):

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = process_image(frame)
        post_process(frame, outputs, conf_threshold)
        
        cv.imshow('Video', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
    
def main():
    
    args = cli_argument_parser()
    
    load_model(args.model_cfg, args.model_weights, args.classes_file)
    
    if args.mode == 'image':
        load_image(args.image_path, args.conf_threshold) 
    elif args.mode == 'video':
        process_video(args.video_path, args.conf_threshold)


if __name__ == '__main__':
    sys.exit(main() or 0)
    
    


