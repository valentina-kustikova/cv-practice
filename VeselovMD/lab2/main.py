import cv2
import numpy as np


def load_model(model_cfg, model_weights):
    """
    Загружает нейронную сеть из файлов конфигурации и весов.
    """
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_output_layers(net):
    """
    Возвращает имена выходных слоев модели.
    """
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def detect_objects(image, net, output_layers, classes, confidence_threshold=0.6, nms_threshold=0.3):
    """
    Выполняет детектирование объектов на изображении.

    Параметры:
        image: входное изображение
        net: модель для детектирования
        output_layers: выходные слои модели
        classes: список классов объектов
        confidence_threshold: порог уверенности
        nms_threshold: порог NMS (подавления незначимых рамок)

    Возвращает:
        list: обнаруженные объекты в формате (x, y, w, h, class_id, confidence).
    """
    # Пред обработка
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Извлечение данных
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Рисуем рамки
    result_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            result_boxes.append((x, y, w, h, class_ids[i], confidences[i]))
    return result_boxes


def draw_predictions(image, detections, classes):
    """
    Отображает предсказания на изображении.
    """
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for (x, y, w, h, class_id, confidence) in detections:
        label = f"{classes[class_id]} {confidence:.3f}"
        color = colors[class_id]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def main():
    # Пути к файлам модели и именам классов
    model_cfg = "yolov4.cfg"
    model_weights = "yolov4.weights"
    class_file = "coco.names"

    # Загружаем классы объектов
    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Загружаем модель
    net = load_model(model_cfg, model_weights)
    output_layers = get_output_layers(net)

    # Загружаем изображение
    image_path = "test_image_2.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image '{image_path}'")
        return

    # Выполняем детектирование
    detections = detect_objects(image, net, output_layers, classes)

    # Рисуем результаты
    image_with_predictions = draw_predictions(image, detections, classes)

    # Выводим статистику
    stats = {}
    for _, _, _, _, class_id, _ in detections:
        class_name = classes[class_id]
        stats[class_name] = stats.get(class_name, 0) + 1

    for class_name, count in stats.items():
        print(f"{class_name}: {count}")

    # Отображаем изображение
    # cv2.imshow("Detections", image_with_predictions)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Сохраняем изображение
    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, image_with_predictions)
    print(f"Image saved to {output_image_path}")


if __name__ == "__main__":
    main()
