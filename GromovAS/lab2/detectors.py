"""
Модуль реализации детекторов объектов на основе нейронных сетей.
Поддерживает различные архитектуры для распознавания транспортных средств.
"""

import cv2
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class ObjectDetectionModel(ABC):
    """Абстрактный базовый класс для детекторов объектов"""

    def __init__(self,
                 model_config: str,
                 model_weights: str,
                 class_labels_path: Optional[str] = None,
                 confidence_limit: float = 0.5,
                 nms_limit: float = 0.4):
        """
        Инициализация детектора.

        Args:
            model_config: Путь к файлу конфигурации модели
            model_weights: Путь к файлу весов модели
            class_labels_path: Путь к файлу с метками классов
            confidence_limit: Порог уверенности для фильтрации
            nms_limit: Порог для подавления немаксимумов
        """
        # Загрузка модели
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Файл весов не найден: {model_weights}")

        self.neural_net = cv2.dnn.readNet(model_weights, model_config)
        self.confidence_threshold = confidence_limit
        self.nms_threshold = nms_limit
        self.class_names = []

        # Загрузка меток классов
        if class_labels_path and os.path.exists(class_labels_path):
            with open(class_labels_path, 'r') as labels_file:
                self.class_names = [line.strip() for line in labels_file]

        # Генерация цветов для отображения классов
        np.random.seed(12345)
        self.class_colors = np.random.uniform(0, 255, size=(len(self.class_names) or 100, 3))

    @abstractmethod
    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Подготавливает изображение для подачи в нейронную сеть.

        Args:
            image: Исходное изображение в формате BGR

        Returns:
            Обработанный blob
        """
        pass

    @abstractmethod
    def process_output(self,
                       image: np.ndarray,
                       network_outputs: List[np.ndarray]) -> List[List]:
        """
        Обрабатывает выход нейронной сети и извлекает обнаружения.

        Args:
            image: Исходное изображение
            network_outputs: Выходы сети

        Returns:
            Список обнаружений [класс, уверенность, x, y, ширина, высота]
        """
        pass

    def detect_objects(self, image: np.ndarray) -> List[List]:
        """
        Основной метод для обнаружения объектов на изображении.

        Args:
            image: Входное изображение

        Returns:
            Список обнаруженных объектов
        """
        # Предобработка
        input_blob = self.prepare_image(image)

        # Прямой проход через сеть
        self.neural_net.setInput(input_blob)
        output_layers = self.neural_net.getUnconnectedOutLayersNames()
        raw_output = self.neural_net.forward(output_layers)

        # Постобработка
        return self.process_output(image, raw_output)

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Возвращает цвет для отрисовки рамки класса.

        Args:
            class_name: Название класса

        Returns:
            Цвет в формате BGR
        """
        if not self.class_names:
            return (0, 255, 0)  # Зеленый по умолчанию

        try:
            idx = self.class_names.index(class_name) % len(self.class_colors)
        except ValueError:
            # Если класс не найден, используем хэш
            idx = hash(class_name) % len(self.class_colors)

        color = self.class_colors[idx]
        return (int(color[0]), int(color[1]), int(color[2]))

    def set_confidence_threshold(self, threshold: float) -> None:
        """Устанавливает новый порог уверенности"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))


class SSDMobileNetDetector(ObjectDetectionModel):
    """Детектор на основе архитектуры SSD MobileNet"""

    def __init__(self, model_config: str, model_weights: str):
        # VOC классы по умолчанию
        default_classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

        super().__init__(model_config, model_weights)

        # Используем стандартные классы, если не загружены из файла
        if not self.class_names:
            self.class_names = default_classes

        # Индексы транспортных средств в VOC
        self.vehicle_indices = {6: "bus", 7: "car", 14: "motorbike", 19: "train"}

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Подготовка изображения для SSD MobileNet.

        Args:
            image: Исходное изображение

        Returns:
            Нормализованный blob
        """
        # Параметры для SSD MobileNet
        target_size = (300, 300)
        scale_factor = 0.007843
        mean_values = (127.5, 127.5, 127.5)

        return cv2.dnn.blobFromImage(
            image,
            scalefactor=scale_factor,
            size=target_size,
            mean=mean_values,
            swapRB=False
        )

    def process_output(self,
                       image: np.ndarray,
                       network_outputs: List[np.ndarray]) -> List[List]:
        """
        Обработка выхода SSD сети.

        Args:
            image: Исходное изображение
            network_outputs: Выход нейронной сети

        Returns:
            Отфильтрованные обнаружения
        """
        image_height, image_width = image.shape[:2]
        detections = []

        # Извлекаем детекции
        detection_data = network_outputs[0]

        for detection_idx in range(detection_data.shape[2]):
            detection_confidence = detection_data[0, 0, detection_idx, 2]

            # Фильтрация по порогу уверенности
            if detection_confidence <= self.confidence_threshold:
                continue

            # Определение класса
            class_index = int(detection_data[0, 0, detection_idx, 1])

            # Фильтрация по классу (только транспорт)
            if class_index not in self.vehicle_indices:
                continue

            # Извлечение координат
            box_coords = detection_data[0, 0, detection_idx, 3:7]
            box_coords = box_coords * np.array([
                image_width,
                image_height,
                image_width,
                image_height
            ])

            start_x, start_y, end_x, end_y = box_coords.astype("int")

            # Корректировка координат
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(image_width, end_x)
            end_y = min(image_height, end_y)

            width = end_x - start_x
            height = end_y - start_y

            if width <= 0 or height <= 0:
                continue

            # Получаем название класса
            class_name = self.vehicle_indices.get(
                class_index,
                self.class_names[class_index] if class_index < len(self.class_names) else "unknown"
            )

            detections.append([
                class_name,
                float(detection_confidence),
                start_x,
                start_y,
                width,
                height
            ])

        return detections


class YOLOv3Detector(ObjectDetectionModel):
    """Детектор на основе архитектуры YOLOv3"""

    def __init__(self, model_config: str, model_weights: str, class_labels_path: str):
        super().__init__(model_config, model_weights, class_labels_path)

        # Индексы транспортных средств в COCO
        self.transport_class_ids = {2, 3, 5, 6, 7}  # car, motorbike, bus, train, truck

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Подготовка изображения для YOLOv3.

        Args:
            image: Исходное изображение

        Returns:
            Нормализованный blob
        """
        target_size = (416, 416)
        scale_factor = 1.0 / 255.0

        return cv2.dnn.blobFromImage(
            image,
            scalefactor=scale_factor,
            size=target_size,
            swapRB=True,
            crop=False
        )

    def process_output(self,
                       image: np.ndarray,
                       network_outputs: List[np.ndarray]) -> List[List]:
        """
        Обработка выхода YOLOv3 сети с применением NMS.

        Args:
            image: Исходное изображение
            network_outputs: Выходы нейронной сети

        Returns:
            Отфильтрованные обнаружения
        """
        image_height, image_width = image.shape[:2]
        bounding_boxes = []
        confidence_scores = []
        class_indices = []

        # Обработка каждого выхода сети
        for layer_output in network_outputs:
            for detection in layer_output:
                # Извлекаем вероятности классов
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Фильтрация по уверенности и классу
                if (confidence > self.confidence_threshold and
                        class_id in self.transport_class_ids):
                    # Конвертация координат
                    box = detection[0:4] * np.array([
                        image_width,
                        image_height,
                        image_width,
                        image_height
                    ])

                    center_x, center_y, box_width, box_height = box.astype("int")

                    # Конвертация из центра к углу
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))

                    bounding_boxes.append([x, y, int(box_width), int(box_height)])
                    confidence_scores.append(float(confidence))
                    class_indices.append(class_id)

        # Применение Non-Maximum Suppression для удаления дубликатов
        if bounding_boxes:
            nms_indices = cv2.dnn.NMSBoxes(
                bounding_boxes,
                confidence_scores,
                self.confidence_threshold,
                self.nms_threshold
            )
        else:
            return []

        final_detections = []

        if len(nms_indices) > 0:
            # Для OpenCV разных версий
            if hasattr(nms_indices, 'flatten'):
                nms_indices = nms_indices.flatten()

            for idx in nms_indices:
                class_id = class_indices[idx]

                # Получаем название класса
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"

                x, y, w, h = bounding_boxes[idx]

                # Корректировка координат
                x = max(0, x)
                y = max(0, y)
                w = min(w, image_width - x)
                h = min(h, image_height - y)

                if w <= 0 or h <= 0:
                    continue

                final_detections.append([
                    class_name,
                    confidence_scores[idx],
                    x,
                    y,
                    w,
                    h
                ])

        return final_detections