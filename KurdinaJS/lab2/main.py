import argparse
import sys
import cv2 as cv
import numpy as np

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
                        help='Path to a video file',
                        type=str,
                        dest='video_path')
    parser.add_argument('-cfg', '--config',
                        help='Path to the YOLO configuration file',
                        type=str,
                        dest='config_path')
    parser.add_argument('-w', '--weights',
                        help='Path to the YOLO weights file',
                        type=str,
                        dest='weights_path')
    parser.add_argument('-n', '--names',
                        help='Path to the file with the class names',
                        type=str,
                        dest='names_path')
    parser.add_argument('-conf', '--confidence_threshold',
                        help="Confidence threshold",
                        type=float,
                        default=0.5)
    parser.add_argument('-nms', '--nms_threshold',
                        help="NMS threshold",
                        type=float,
                        default=0.4)

    args = parser.parse_args()
    return args


def get_colors_and_classes(names_path):
    with open(names_path, 'r') as f:
        names = []
        for line in f.readlines():
            names.append(line.strip())

    colors = np.random.uniform(0, 255, size=(len(names), 3))
    return colors, names


def extract_bounding_box(detection, height, width):
    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
    obj_width, obj_height = int(detection[2] * width), int(detection[3] * height)
    top_left_x = int(center_x - obj_width / 2)
    top_left_y = int(center_y - obj_height / 2)
    return top_left_x, top_left_y, obj_width, obj_height


def load_model(weights_path, config_path, names_path):
    model = cv.dnn.readNet(weights_path, config_path)
    colors, names = get_colors_and_classes(names_path)
    return model, names, colors


def draw_predictions(image, nms_indices, bounding_boxes, class_labels, colors, detection_confidences, detected_classes):
    obj_count = {}
    for idx in nms_indices:
        x, y, width, height = bounding_boxes[idx]
        color = colors[detected_classes[idx]]

        cv.rectangle(image, (x, y), (x + width, y + height), color, 2)
        cv.putText(image, f"{class_labels[detected_classes[idx]]}: {detection_confidences[idx]:.3f}",
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        object_name = class_labels[detected_classes[idx]]
        obj_count[object_name] = obj_count.get(object_name, 0) + 1

    return obj_count


def detect_and_display_objects(image, outputs, class_labels, colors, height, width, conf_threshold, nms_threshold):
    bounding_boxes = []
    detection_confidences = []
    detected_classes = []

    for detection_layer in outputs:
        for detection in detection_layer:
            class_probabilities = detection[5:]
            predicted_class = np.argmax(class_probabilities)
            confidence_score = class_probabilities[predicted_class]

            if confidence_score > conf_threshold:
                top_left_x, top_left_y, obj_width, obj_height = extract_bounding_box(detection, height, width)

                bounding_boxes.append([top_left_x, top_left_y, obj_width, obj_height])
                detection_confidences.append(confidence_score)
                detected_classes.append(predicted_class)

    nms_indices = cv.dnn.NMSBoxes(bounding_boxes, detection_confidences, conf_threshold, nms_threshold)
    nms_indices.flatten()

    obj_count = draw_predictions(image, nms_indices, bounding_boxes, class_labels, colors, detection_confidences,
                                 detected_classes)

    return image, obj_count


def image_processing(image, model, classes, colors, confidence_threshold, nms_threshold):
    height, width = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)

    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(output_layers)

    result_image, detected_obj = detect_and_display_objects(
        image, outputs, classes, colors, height, width, confidence_threshold, nms_threshold
    )

    return result_image, detected_obj


def objectDetection_image(image_path, model, classes, colors, confidence_threshold, nms_threshold):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError('Empty path to the image')

    result_image, detected_obj = image_processing(image, model, classes, colors, confidence_threshold, nms_threshold)

    print(f"The image contains:")
    for obj, count in detected_obj.items():
        print(f"{obj}: {count}")

    cv.imshow(f"{image_path}", result_image)
    key = cv.waitKey(0) & 0xFF
    if key == ord('q'):
        cv.destroyAllWindows()


def objectDetection_video(video_path, model, classes, colors, confidence_threshold, nms_threshold):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Empty path to the video")
    k = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, detected_obj = image_processing(frame, model, classes, colors, confidence_threshold,
                                                      nms_threshold)
        k += 1
        print(f"Frame: {k}")
        for obj, count in detected_obj.items():
            print(f"{obj}: {count}")

        cv.imshow(f"{video_path}", result_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def main():
    args = cli_argument_parser()
    model, classes, colors = load_model(args.weights_path, args.config_path, args.names_path)

    if args.mode == 'image' and args.image_path:
        objectDetection_image(args.image_path, model, classes, colors, args.confidence_threshold, args.nms_threshold)
    elif args.mode == 'video' and args.video_path:
        objectDetection_video(args.video_path, model, classes, colors, args.confidence_threshold, args.nms_threshold)
    else:
        raise ValueError("Unsupported 'mode' value or missing path")


if __name__ == '__main__':
    sys.exit(main() or 0)
