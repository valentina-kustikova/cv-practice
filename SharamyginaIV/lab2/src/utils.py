import cv2
import numpy as np
import os


def draw_predictions(image, boxes, class_ids, confidences, classes, colors):
    output_image = image.copy()
    height, width = image.shape[:2]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        class_id = class_ids[i]
        confidence = confidences[i]

        label = f"{classes[class_id]}: {confidence:.3f}"
        color = colors[class_id]

        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_width = label_size[0]

        label_x = min(x1, width - label_width)

        label_ymin = max(y1, label_size[1] + 10)

        cv2.rectangle(output_image, (label_x, label_ymin - label_size[1] - 10), (label_x + label_width, label_ymin + 5),
                      color, cv2.FILLED)
        cv2.putText(output_image, label, (label_x, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return output_image

def load_ground_truth_labels(filepath):
    class_name_to_id = {
        "CAR": 2,
        "BUS": 5,
    }

    labels = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 6:
                frame_number = int(parts[0])
                class_name = parts[1] # "CAR" или "BUS"
                x1 = int(parts[2])
                y1 = int(parts[3])
                x2 = int(parts[4])
                y2 = int(parts[5])
                filename = f"{frame_number:06d}.jpg"
                if filename not in labels:
                    labels[filename] = []
                if class_name in class_name_to_id:
                    class_id = class_name_to_id[class_name]
                    labels[filename].append({'bbox': [x1, y1, x2, y2], 'class': class_id})
                else:
                    print(f"Warning: Unknown class '{class_name}' in ground truth for frame {filename}. Skipping.")
    return labels

def get_image_files(directory):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]
    image_files.sort() # Сортируем, чтобы кадры шли по порядку
    return image_files
