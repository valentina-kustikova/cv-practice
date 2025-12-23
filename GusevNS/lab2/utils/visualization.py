import cv2


def class_color(label):
    value = abs(hash(label))
    b = (value & 255)
    g = (value >> 8) & 255
    r = (value >> 16) & 255
    return int(b), int(g), int(r)


def draw_detections(image, detections, font_scale=0.5, thickness=1):
    output = image.copy()
    for detection in detections:
        x, y, w, h = detection.box
        color = class_color(detection.label)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        text = f"{detection.label} {detection.confidence:.3f}"
        cv2.putText(output, text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(output, detection.label.upper(), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return output


def draw_truths(image, truths, color=(0, 255, 0), font_scale=0.5, thickness=1):
    output = image.copy()
    for truth in truths:
        x1, y1, x2, y2 = truth.box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        cv2.putText(output, truth.class_name.upper(), (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return output

