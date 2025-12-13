"""
Модуль с иерархией классов детекторов объектов.
Использует модуль cv2.dnn для инференса нейронных сетей.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


@dataclass
class Detection:
    """Класс для хранения результата детекции одного объекта."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class BaseDetector(ABC):
    """
    Абстрактный базовый класс для всех детекторов.
    
    Определяет интерфейс для детекции объектов:
    - preprocess: предобработка изображения
    - postprocess: постобработка выхода сети
    - detect: основной метод детекции
    """
    
    # Классы COCO, которые нас интересуют (только CAR и BUS)
    VEHICLE_CLASSES = {'car', 'bus'}
    
    # Маппинг классов COCO на наши классы
    CLASS_MAPPING = {
        'car': 'CAR',
        'bus': 'BUS',
    }
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Инициализация детектора.
        
        Args:
            model_path: путь к файлу модели
            config_path: путь к файлу конфигурации (опционально)
            confidence_threshold: порог уверенности для детекций
            nms_threshold: порог для Non-Maximum Suppression
        """
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.classes = []
        
        self._load_model()
        self._load_classes()
    
    @abstractmethod
    def _load_model(self) -> None:
        """Загрузка модели. Реализуется в подклассах."""
        pass
    
    @abstractmethod
    def _load_classes(self) -> None:
        """Загрузка списка классов. Реализуется в подклассах."""
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения перед подачей в сеть.
        
        Args:
            image: входное изображение в формате BGR (OpenCV)
            
        Returns:
            blob: подготовленный тензор для сети
        """
        pass
    
    @abstractmethod
    def postprocess(self, 
                    outputs: List[np.ndarray], 
                    image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Постобработка выхода сети.
        
        Args:
            outputs: выходы нейронной сети
            image_shape: размер исходного изображения (height, width)
            
        Returns:
            detections: список детекций
        """
        pass
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Основной метод детекции объектов на изображении.
        
        Args:
            image: входное изображение в формате BGR
            
        Returns:
            detections: список детекций транспортных средств
        """
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward(self._get_output_layers())
        detections = self.postprocess(outputs, image.shape[:2])
        
        # Фильтруем только транспортные средства
        vehicle_detections = []
        for det in detections:
            if det.class_name.lower() in self.VEHICLE_CLASSES:
                # Маппим на наши классы
                mapped_class = self.CLASS_MAPPING.get(det.class_name.lower(), det.class_name.upper())
                det.class_name = mapped_class
                vehicle_detections.append(det)
        
        return vehicle_detections
    
    @abstractmethod
    def _get_output_layers(self) -> List[str]:
        """Возвращает имена выходных слоёв сети."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Возвращает название детектора."""
        pass


class YOLOv8Detector(BaseDetector):
    """
    Детектор на основе YOLOv8 (ONNX формат).
    
    Особенности предобработки:
    - Изменение размера до 640x640 с сохранением пропорций (letterbox)
    - Нормализация значений пикселей в диапазон [0, 1]
    - Конвертация BGR -> RGB
    
    Особенности постобработки:
    - Выход сети: [1, 84, 8400] где 84 = 4 (bbox) + 80 (классы COCO)
    - Транспонирование для удобства обработки
    - Применение NMS для удаления дублирующих детекций
    """
    
    # Классы COCO (80 классов)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    INPUT_SIZE = (640, 640)
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        super().__init__(model_path, None, confidence_threshold, nms_threshold)
    
    def _load_model(self) -> None:
        """Загрузка ONNX модели YOLOv8."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        # Используем CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def _load_classes(self) -> None:
        """Загрузка классов COCO."""
        self.classes = self.COCO_CLASSES
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка для YOLOv8:
        1. Letterbox resize до 640x640 с сохранением пропорций
        2. BGR -> RGB
        3. Нормализация [0, 255] -> [0, 1]
        4. Транспонирование HWC -> CHW
        5. Добавление batch dimension
        """
        # Сохраняем оригинальные размеры для постобработки
        self.original_shape = image.shape[:2]
        
        # Letterbox resize
        h, w = image.shape[:2]
        target_h, target_w = self.INPUT_SIZE
        
        # Вычисляем коэффициент масштабирования
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Масштабируем изображение
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Создаём letterbox (серый фон)
        letterbox = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Вычисляем отступы
        self.pad_x = (target_w - new_w) // 2
        self.pad_y = (target_h - new_h) // 2
        self.scale = scale
        
        # Помещаем изображение в центр
        letterbox[self.pad_y:self.pad_y + new_h, self.pad_x:self.pad_x + new_w] = resized
        
        # BGR -> RGB и нормализация
        blob = cv2.dnn.blobFromImage(
            letterbox,
            scalefactor=1/255.0,
            size=self.INPUT_SIZE,
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        return blob
    
    def postprocess(self, outputs: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Постобработка выхода YOLOv8:
        1. Транспонирование выхода [1, 84, 8400] -> [8400, 84]
        2. Извлечение bbox и scores
        3. Фильтрация по confidence
        4. Применение NMS
        5. Преобразование координат обратно в исходное изображение
        """
        detections = []
        output = outputs[0]
        
        # YOLOv8 output shape: [1, 84, 8400] -> транспонируем в [8400, 84]
        if len(output.shape) == 3:
            output = output[0]
        output = output.T  # [8400, 84]
        
        boxes = []
        confidences = []
        class_ids = []
        
        h, w = image_shape
        
        for detection in output:
            # Первые 4 значения - bbox (cx, cy, w, h)
            cx, cy, bw, bh = detection[:4]
            
            # Остальные 80 - scores для каждого класса
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Преобразуем координаты из letterbox в оригинальное изображение
            # Сначала из центра в углы
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            
            # Убираем padding
            x1 = (x1 - self.pad_x) / self.scale
            y1 = (y1 - self.pad_y) / self.scale
            x2 = (x2 - self.pad_x) / self.scale
            y2 = (y2 - self.pad_y) / self.scale
            
            # Клиппинг
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        
        # Применяем NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                x, y, bw, bh = boxes[idx]
                detections.append(Detection(
                    class_id=class_ids[idx],
                    class_name=self.classes[class_ids[idx]],
                    confidence=confidences[idx],
                    x1=x,
                    y1=y,
                    x2=x + bw,
                    y2=y + bh
                ))
        
        return detections
    
    def _get_output_layers(self) -> List[str]:
        """YOLOv8 ONNX имеет один выходной слой."""
        layer_names = self.net.getLayerNames()
        output_layers = self.net.getUnconnectedOutLayers()
        return [layer_names[i - 1] for i in output_layers.flatten()]
    
    @property
    def name(self) -> str:
        return "YOLOv8l"


class SSDMobileNetDetector(BaseDetector):
    """
    Детектор на основе SSD MobileNet (Caffe).
    
    Особенности предобработки:
    - Изменение размера до 300x300
    - Вычитание среднего: mean=[127.5, 127.5, 127.5]
    - Масштабирование: scale=1/127.5=0.007843
    
    Особенности постобработки:
    - Выход сети: [1, 1, N, 7] где 7 = [batch_id, class_id, confidence, x1, y1, x2, y2]
    - Координаты нормализованы [0, 1]
    - NMS уже применён внутри модели
    """
    
    # Классы COCO (90 классов для TF Object Detection API) или VOC (21 класс)
    COCO_CLASSES_TF = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
        'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    INPUT_SIZE = (300, 300)
    
    def __init__(self, model_path: str, config_path: str, 
                 confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        super().__init__(model_path, config_path, confidence_threshold, nms_threshold)
    
    def _load_model(self) -> None:
        """Загрузка модели SSD (поддержка Caffe и TensorFlow форматов)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        if self.config_path and not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {self.config_path}")
        
        # Определяем формат по расширению файла
        if self.model_path.endswith('.caffemodel'):
            self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
            self.is_caffe = True
        else:
            self.net = cv2.dnn.readNetFromTensorflow(self.model_path, self.config_path)
            self.is_caffe = False
            
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def _load_classes(self) -> None:
        """Загрузка классов COCO для TF Object Detection API."""
        self.classes = self.COCO_CLASSES_TF
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка для SSD MobileNet:
        1. Resize до 300x300
        2. Нормализация: mean=[127.5, 127.5, 127.5], scale=1/127.5
        """
        self.original_shape = image.shape[:2]
        
        # Для Caffe MobileNet-SSD используем специфичную нормализацию
        if hasattr(self, 'is_caffe') and self.is_caffe:
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=0.007843,  # 1/127.5
                size=self.INPUT_SIZE,
                mean=(127.5, 127.5, 127.5),
                swapRB=False,
                crop=False
            )
        else:
            # Для TensorFlow модели
            blob = cv2.dnn.blobFromImage(
                image,
                scalefactor=1.0,
                size=self.INPUT_SIZE,
                mean=(0, 0, 0),
                swapRB=True,
                crop=False
            )
        
        return blob
    
    def postprocess(self, outputs: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Постобработка выхода SSD:
        1. Парсинг выхода [1, 1, N, 7]
        2. Фильтрация по confidence
        3. Денормализация координат
        """
        detections = []
        output = outputs[0]
        
        h, w = image_shape
        
        # Output shape: [1, 1, num_detections, 7]
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            
            if confidence < self.confidence_threshold:
                continue
            
            class_id = int(detection[1])
            
            # Координаты нормализованы [0, 1]
            x1 = int(detection[3] * w)
            y1 = int(detection[4] * h)
            x2 = int(detection[5] * w)
            y2 = int(detection[6] * h)
            
            # Клиппинг
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            if class_id < len(self.classes):
                class_name = self.classes[class_id]
            else:
                class_name = f"class_{class_id}"
            
            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            ))
        
        return detections
    
    def _get_output_layers(self) -> List[str]:
        """SSD TensorFlow имеет стандартный выход."""
        layer_names = self.net.getLayerNames()
        output_layers = self.net.getUnconnectedOutLayers()
        return [layer_names[i - 1] for i in output_layers.flatten()]
    
    @property
    def name(self) -> str:
        if hasattr(self, 'is_caffe') and self.is_caffe:
            return "SSD MobileNet (Caffe)"
        return "SSD MobileNet"


class NanoDetPlusDetector(BaseDetector):
    """
    Детектор на основе NanoDet-Plus (ONNX формат).
    
    Особенности предобработки:
    - Изменение размера до 416x416 с сохранением пропорций
    - Нормализация mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]
    - BGR порядок (без swapRB)
    
    Особенности постобработки:
    - Выход сети: [1, 3598, 112] - три уровня FPN с разными stride
    - 112 = 80 классов + 32 (distribution для bbox)
    - Требуется декодирование GFL (Generalized Focal Loss)
    - Применение NMS
    """
    
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    INPUT_SIZE = (416, 416)
    REG_MAX = 7  # Для GFL декодирования
    STRIDES = [8, 16, 32]  # Stride для каждого уровня FPN
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        super().__init__(model_path, None, confidence_threshold, nms_threshold)
        self._generate_anchors()
    
    def _generate_anchors(self, output_size: int = None) -> None:
        """Генерация anchor points для каждого уровня FPN."""
        self.anchor_points = []
        self.stride_tensor = []
        
        for stride in self.STRIDES:
            h = self.INPUT_SIZE[0] // stride
            w = self.INPUT_SIZE[1] // stride
            
            # Создаём сетку anchor points
            shift_x = np.arange(0, w) + 0.5
            shift_y = np.arange(0, h) + 0.5
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            
            points = np.stack([shift_x.flatten(), shift_y.flatten()], axis=-1) * stride
            self.anchor_points.append(points)
            self.stride_tensor.append(np.full((h * w,), stride))
        
        self.anchor_points = np.concatenate(self.anchor_points, axis=0)
        self.stride_tensor = np.concatenate(self.stride_tensor, axis=0)
        
        # Если размер выхода отличается, обрезаем или дополняем
        if output_size is not None and output_size != len(self.anchor_points):
            if output_size < len(self.anchor_points):
                self.anchor_points = self.anchor_points[:output_size]
                self.stride_tensor = self.stride_tensor[:output_size]
            else:
                # Дополняем последними значениями
                diff = output_size - len(self.anchor_points)
                self.anchor_points = np.concatenate([
                    self.anchor_points,
                    np.tile(self.anchor_points[-1:], (diff, 1))
                ], axis=0)
                self.stride_tensor = np.concatenate([
                    self.stride_tensor,
                    np.full((diff,), self.STRIDES[-1])
                ], axis=0)
    
    def _load_model(self) -> None:
        """Загрузка ONNX модели NanoDet-Plus."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def _load_classes(self) -> None:
        """Загрузка классов COCO."""
        self.classes = self.COCO_CLASSES
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка для NanoDet-Plus:
        1. Letterbox resize до 416x416
        2. Нормализация с mean и std
        """
        self.original_shape = image.shape[:2]
        h, w = image.shape[:2]
        target_h, target_w = self.INPUT_SIZE
        
        # Letterbox
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        letterbox = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
        self.pad_x = (target_w - new_w) // 2
        self.pad_y = (target_h - new_h) // 2
        self.scale = scale
        
        letterbox[self.pad_y:self.pad_y + new_h, self.pad_x:self.pad_x + new_w] = resized
        
        # Нормализация: mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]
        mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
        std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
        
        blob = letterbox.astype(np.float32)
        blob = (blob - mean) / std
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # Добавляем batch dimension
        
        return blob
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Вычисление softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _decode_gfl(self, bbox_pred: np.ndarray) -> np.ndarray:
        """
        Декодирование GFL (Generalized Focal Loss) предсказаний bbox.
        
        Args:
            bbox_pred: [N, 4 * (reg_max + 1)] - распределение для каждой стороны bbox
            
        Returns:
            decoded: [N, 4] - декодированные смещения (left, top, right, bottom)
        """
        # Reshape для softmax по каждой стороне
        bbox_pred = bbox_pred.reshape(-1, 4, self.REG_MAX + 1)
        
        # Применяем softmax
        bbox_pred = self._softmax(bbox_pred, axis=-1)
        
        # Вычисляем взвешенную сумму (expectation)
        proj = np.arange(self.REG_MAX + 1, dtype=np.float32)
        decoded = np.sum(bbox_pred * proj, axis=-1)
        
        return decoded
    
    def postprocess(self, outputs: List[np.ndarray], image_shape: Tuple[int, int]) -> List[Detection]:
        """
        Постобработка выхода NanoDet-Plus:
        1. Разделение на cls_scores и bbox_preds
        2. Декодирование GFL для bbox
        3. Фильтрация по confidence
        4. Преобразование координат
        5. Применение NMS
        """
        detections = []
        output = outputs[0][0]  # [N, 112]
        
        # Проверяем соответствие размера выхода и anchor points
        output_size = len(output)
        if len(self.anchor_points) != output_size:
            self._generate_anchors(output_size)
        
        h, w = image_shape
        num_classes = len(self.classes)
        
        # Разделяем на классы и bbox
        cls_scores = output[:, :num_classes]  # [N, 80]
        bbox_preds = output[:, num_classes:]  # [3598, 32]
        
        # Декодируем bbox
        bbox_decoded = self._decode_gfl(bbox_preds)  # [3598, 4] - left, top, right, bottom
        
        boxes = []
        confidences = []
        class_ids = []
        
        for i in range(len(cls_scores)):
            scores = cls_scores[i]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Получаем смещения
            left, top, right, bottom = bbox_decoded[i] * self.stride_tensor[i]
            
            # Преобразуем в координаты
            cx, cy = self.anchor_points[i]
            x1 = cx - left
            y1 = cy - top
            x2 = cx + right
            y2 = cy + bottom
            
            # Убираем letterbox padding и масштабирование
            x1 = (x1 - self.pad_x) / self.scale
            y1 = (y1 - self.pad_y) / self.scale
            x2 = (x2 - self.pad_x) / self.scale
            y2 = (y2 - self.pad_y) / self.scale
            
            # Клиппинг
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        
        # Применяем NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                x, y, bw, bh = boxes[idx]
                detections.append(Detection(
                    class_id=class_ids[idx],
                    class_name=self.classes[class_ids[idx]],
                    confidence=confidences[idx],
                    x1=x,
                    y1=y,
                    x2=x + bw,
                    y2=y + bh
                ))
        
        return detections
    
    def _get_output_layers(self) -> List[str]:
        """NanoDet-Plus ONNX выходной слой."""
        layer_names = self.net.getLayerNames()
        output_layers = self.net.getUnconnectedOutLayers()
        return [layer_names[i - 1] for i in output_layers.flatten()]
    
    @property
    def name(self) -> str:
        return "NanoDet-Plus"


def create_detector(model_name: str, 
                    model_path: str, 
                    config_path: Optional[str] = None,
                    confidence_threshold: float = 0.5,
                    nms_threshold: float = 0.4) -> BaseDetector:
    """
    Фабричный метод для создания детектора.
    
    Args:
        model_name: название модели ('yolov8', 'ssd', 'nanodet')
        model_path: путь к файлу модели
        config_path: путь к файлу конфигурации (для SSD)
        confidence_threshold: порог уверенности
        nms_threshold: порог NMS
        
    Returns:
        detector: экземпляр детектора
    """
    model_name = model_name.lower()
    
    if model_name in ['yolov8', 'yolov8l']:
        return YOLOv8Detector(model_path, confidence_threshold, nms_threshold)
    elif model_name in ['ssd', 'ssd_mobilenet', 'ssd-mobilenet']:
        if config_path is None:
            raise ValueError("Для SSD требуется указать config_path")
        return SSDMobileNetDetector(model_path, config_path, confidence_threshold, nms_threshold)
    elif model_name in ['nanodet', 'nanodet-plus', 'nanodet_plus']:
        return NanoDetPlusDetector(model_path, confidence_threshold, nms_threshold)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}. "
                        f"Доступные модели: yolov8, ssd, nanodet")
