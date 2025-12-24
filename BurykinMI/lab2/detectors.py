import cv2
import numpy as np
import os
from abc import ABC, abstractmethod


# ============================================================================
# Иерархия классов детекторов (Model Layer)
# ============================================================================

class BaseDetector(ABC):
    def __init__(self, config_path, weights_path, classes_path=None, conf_threshold=0.5, nms_threshold=0.4):
        # cv2.dnn.readNet сама понимает, Caffe это или Darknet, по расширению файлов
        self.net = cv2.dnn.readNet(weights_path, config_path)

        # Включаем использование CPU (можно поменять на CUDA, если есть GPU)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold  # Порог перекрытия для NMS (удаление дублей)
        self.classes = []

        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

        # Генерируем палитру (фиксированный seed, чтобы цвета классов не менялись при перезапуске)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    @abstractmethod
    def preprocess(self, image):
        """
        Превращает (H, W, 3) изображение в 4D-тензор (Blob) формата NCHW.
        """
        pass

    @abstractmethod
    def postprocess(self, image, outputs):
        """
        Парсит выходные тензоры сети в список координат [label, conf, x, y, w, h].
        """
        pass

    def detect(self, image):
        # 1. Подготовка: Image -> Blob (1, 3, H, W)
        blob = self.preprocess(image)
        self.net.setInput(blob)

        # 2. Инференс: Прогон через слои
        # getUnconnectedOutLayersNames() возвращает имена всех выходных слоев (1 для SSD, 3 для YOLOv3, 2 для Tiny)
        output_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(output_names)

        # 3. Декодирование результатов
        return self.postprocess(image, outputs)

    def get_color(self, label_str):
        idx = hash(label_str) % 100
        return [int(c) for c in self.colors[idx]]


class SSDDetector(BaseDetector):
    def __init__(self, config_path, weights_path, classes_path=None):
        super().__init__(config_path, weights_path, classes_path)
        # Стандартные классы VOC, если файл не передан (для MobileNet-SSD)
        if not self.classes:
            self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                            "dog", "horse", "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"]

    def preprocess(self, image):
        # SSD требует вход 300x300.
        # Формула: (Pixel - Mean) * Scale
        # Scale = 1/127.5 ≈ 0.007843. Mean = 127.5.
        # Итог: значения пикселей приводятся к диапазону [-1, 1].
        # SwapRB=False, т.к. SSD обучалась на BGR.
        return cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    def postprocess(self, image, outputs):
        (h, w) = image.shape[:2]

        # Выход SSD — это один тензор формы (1, 1, N, 7)
        # N — количество потенциальных детекций (обычно 100)
        detections = outputs[0]
        results = []

        # detections.shape[2] пробегает по всем N боксам
        for i in range(detections.shape[2]):
            # Структура вектора (7 чисел): [batch, class_id, confidence, x1, y1, x2, y2]
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                idx = int(detections[0, 0, i, 1])

                # Фильтр VOC IDs: bus(6), car(7). Остальное игнорируем.
                if idx not in [6, 7]:
                    continue

                # Координаты приходят нормализованными (0..1), умножаем на размеры кадра
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Клиппинг (чтобы рамка не вылезла за пределы картинки)
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
        self.input_size = input_size  # Позволяет менять разрешение (320, 416, 608)

    def preprocess(self, image):
        # YOLO требует нормализации в диапазон [0, 1], поэтому Scale = 1/255.
        # SwapRB=True обязателен: OpenCV читает BGR, а YOLO обучена на RGB.
        # Используем self.input_size для гибкости (важно для v4-tiny).
        return cv2.dnn.blobFromImage(image, 1 / 255.0, self.input_size, swapRB=True, crop=False)

    def postprocess(self, image, outputs):
        (H, W) = image.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        # YOLO возвращает СПИСОК тензоров (3 шт для v3, 2 шт для v4-tiny).
        # Каждый тензор — это сетка предсказаний разного масштаба.
        for output in outputs:
            # output.shape = (K, 85), где K - кол-во ячеек * якоря
            for detection in output:
                # Вектор (85): [cx, cy, w, h, obj_conf, class_score_1, ... class_score_80]
                scores = detection[5:]  # Вероятности классов
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.conf_threshold:
                    # Фильтр COCO IDs: 2=car, 5=bus, 7=truck
                    if classID not in [2, 5, 7]:
                        continue

                    # Координаты YOLO — это ЦЕНТР бокса и размеры (0..1)
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Перевод в верхний левый угол для OpenCV
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # NMS (Non-Maximum Suppression):
        # YOLO (в отличие от SSD) выдает тысячи перекрывающихся рамок.
        # Эта функция оставляет только лучшие рамки с IoU < nms_threshold.
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                cid = classIDs[i]

                # Маппинг классов COCO в общие категории (Truck -> Car для совместимости с GT)
                if cid == 7:
                    label_str = "car"  # Truck
                elif cid == 2:
                    label_str = "car"  # Car
                elif cid == 5:
                    label_str = "bus"  # Bus
                else:
                    label_str = self.classes[cid]

                results.append([label_str, confidences[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]])

        return results


class YOLOv4TinyDetector(YOLODetector):
    """
    YOLOv4-tiny: облегченная версия.
    Отличие: использует 2 выходных слоя вместо 3 (быстрее, но меньше деталей).
    Логика postprocess полностью совпадает с v3 (формат выхода тот же - 85 чисел).
    """

    def __init__(self, config_path, weights_path, classes_path=None):
        super().__init__(config_path, weights_path, classes_path, input_size=(416, 416))