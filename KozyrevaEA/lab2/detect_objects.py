import cv2
import numpy as np
import os
import logging

# Настройка логгирования
logging.basicConfig(filename='object_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка предварительно обученной модели
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')

# Создание папки для сохранения результатов, если она не существует
output_dir = 'result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Создана папка '{output_dir}' для сохранения результатов.")


def detect_objects(image: np.ndarray):
    """
    Функция для обнаружения объектов на изображении.

    :param image: Входное изображение (формат NumPy array)
    :return: Список кортежей (x1, y1, x2, y2, confidence), где (x1, y1) - координаты верхнего левого угла,
             (x2, y2) - координаты нижнего правого угла, confidence - уверенность в детекции объекта.
    """
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 7:
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                results.append((startX, startY, endX, endY, confidence))
    return results


def show_detected_image(image: np.ndarray, image_name: str) -> None:
    """
    Отображает изображение с аннотированными результатами детекции объектов и сохраняет его.

    :param image: Входное изображение (формат NumPy array)
    :param image_name: Название исходного изображения для сохранения результата
    """
    results = detect_objects(image)
    for (startX, startY, endX, endY, confidence) in results:
        label = f"Car: {confidence:.3f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Сохранение результата
    result_path = os.path.join(output_dir, f"detected_{image_name}")
    cv2.imwrite(result_path, image)
    logging.info(f"Изображение с детекцией сохранено: {result_path}")

    # Отображение изображения
    cv2.imshow("Detected Cars", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Пример использования
image_name = 'img.png'
image = cv2.imread(f'imgs/{image_name}')

if image is not None:
    logging.info(f"Обрабатывается изображение: {image_name}")
    show_detected_image(image, image_name)
else:
    logging.error(f"Не удалось загрузить изображение: {image_name}")
