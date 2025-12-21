import cv2
import numpy as np
import os
from abc import ABC, abstractmethod


# ============================================================================
# Иерархия классов детекторов объектов (Model Layer):
# Отвечает за инкапсуляцию работы с DNN модулем OpenCV.
# Скрывает различия в препроцессинге и постпроцессинге разных архитектур (SSD vs YOLO).
# ============================================================================

class BaseDetector(ABC):
    def __init__(self, config_path, weights_path, classes_path=None, conf_threshold=0.5, nms_threshold=0.4):
        # cv2.dnn.readNet автоматически определяет фреймворк (Caffe или Darknet) по расширению файлов
        self.net = cv2.dnn.readNet(weights_path, config_path)
        # Включаем оптимизации
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold  # Порог для Non-Maximum Suppression
        self.classes = []

        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

        # Генерируем фиксированную палитру цветов
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    @abstractmethod
    def preprocess(self, image):
        """
        Превращает картинку в 4D-тензор (Blob).
        Специфично для каждой сети (разные размеры, разная нормализация).
        """
        pass

    @abstractmethod
    def postprocess(self, image, outputs):
        """
        Декодирует сырой выход сети (матрицы вероятностей) в понятные координаты рамок.
        """
        pass

    def detect(self, image):
        """
        Полный цикл инференса:
        1. Preprocess: подготовка блоба
        2. Forward: прогон через слои сети
        3. Postprocess: фильтрация и парсинг результатов
        """
        blob = self.preprocess(image)
        self.net.setInput(blob)

        # Получаем имена всех выходных слоев (у SSD он 1, у YOLO их 3)
        output_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_names)

        return self.postprocess(image, outputs)

    def get_color(self, label_str):
        # Хэширование строки позволяет всегда выдавать один и тот же цвет для одного класса
        idx = hash(label_str) % 100
        return [int(c) for c in self.colors[idx]]


class SSDDetector(BaseDetector):
    def __init__(self, config_path, weights_path, classes_path=None):
        super().__init__(config_path, weights_path, classes_path)
        # Если файл классов не передан, используем стандартный набор PASCAL VOC
        if not self.classes:
            self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"]

    def preprocess(self, image):
        # Scale 0.007843 = 1/127.5. Вместе с вычитанием среднего (127.5) это приводит пиксели в диапазон [-1, 1].
        return cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    def postprocess(self, image, outputs):
        (h, w) = image.shape[:2]
        detections = outputs[0]  # Формат выхода SSD: (1, 1, N, 7), где 7 - вектор параметров детекции
        results = []

        # detections.shape[2] - это количество найденных потенциальных объектов
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # 2-й элемент - уверенность модели

            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])  # 1-й элемент - ID класса

                # Фильтрация по интересующим классам VOC (6=bus, 7=car)
                if idx not in [6, 7]:
                    continue

                # Координаты приходят в диапазоне [0, 1], масштабируем их под размер исходного кадра
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Защита от выхода за границы
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                label_str = self.classes[idx]
                results.append([label_str, confidence, startX, startY, endX - startX, endY - startY])

        return results


class YOLODetector(BaseDetector):
    def __init__(self, config_path, weights_path, classes_path=None, input_size=(416, 416)):
        super().__init__(config_path, weights_path, classes_path)
        self.input_size = input_size

    def preprocess(self, image):
        # Scale 1/255 приводит пиксели в диапазон [0, 1].
        return cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    def postprocess(self, image, outputs):
        (H, W) = image.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        # Проходим по всем 3 выходным слоям
        for output in outputs:
            for detection in output:
                # Первые 5 чисел - координаты и objectness, остальные - вероятности классов
                scores = detection[5:]
                classID = np.argmax(scores)  # Индекс класса с максимальной вероятностью
                confidence = scores[classID]

                if confidence > self.conf_threshold:
                    # Фильтр COCO классов: 2=car, 5=bus, 7=truck
                    if classID not in [2, 5, 7]:
                        continue

                    # YOLO возвращает центр бокса (CenterX, CenterY) и размеры.
                    # Нужно пересчитать в координаты левого верхнего угла для отрисовки.
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                cid = classIDs[i]

                # Объединяем похожие классы (Truck -> Car) для упрощения сравнения с метриками
                if cid == 7:  # Truck
                    label_str = "car"
                elif cid == 2:  # Car
                    label_str = "car"
                elif cid == 5:  # Bus
                    label_str = "bus"
                else:
                    label_str = self.classes[cid]

                results.append([label_str, confidences[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])

        return results


class YOLOv4TinyDetector(YOLODetector):
    """
    Специализированный класс для YOLOv4-Tiny.
    Хотя логика препроцессинга схожа с v3, выделение в отдельный класс
    позволяет гибко менять input_size или параметры якорей в будущем.
    """

    def __init__(self, config_path, weights_path, classes_path=None):
        # Tiny версии часто хорошо работают и на 416x416, но работают быстрее.
        # Можно поставить (320, 320) для еще большей скорости, но упадет точность.
        super().__init__(config_path, weights_path, classes_path, input_size=(416, 416))