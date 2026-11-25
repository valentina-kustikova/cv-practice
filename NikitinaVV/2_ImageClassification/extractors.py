import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_keep_aspect(image, max_side):
    """Уменьшает изображение, если его большая сторона длиннее max_side."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) #INTER_AREA усредняет пиксели и не дает муара


class FeatureExtractor(ABC):
    """Интерфейс извлечения локальных дескрипторов из изображения."""
    name = None

    @staticmethod
    def create(name, **params):
        """Единый шаблон и фабрика для всех методов извлечения признаков """
        if name not in EXTRACTORS:
            raise ValueError(f'Неизвестный метод извлечения признаков: {name}. Доступны: {", ".join(EXTRACTORS)}')
        params = {key: value for key, value in params.items() if value is not None}
        return EXTRACTORS[name](**params)

    @abstractmethod
    def detect_and_compute(self, image):
        """Принять BGR-изображение, вернуть его дескрипторы: матрицу N x D (float32), одна строка на дескриптор."""

    @abstractmethod
    def get_params(self):
        """вернуть словарь параметров конструктора (чтобы сохранить их в файл модели и при загрузке воссоздать точно такой же экстрактор)"""


class ClassicalExtractor(FeatureExtractor):
    """Общая часть классических методов OpenCV: уменьшение с сохранением пропорций, перевод в серый, detectAndCompute.
    база для SIFT и ORB"""
    descriptor_size = None

    def __init__(self, image_size=640, nfeatures=1000):
        if image_size <= 0 or nfeatures <= 0:
            raise ValueError('image_size и nfeatures должны быть положительными')
        self.image_size = image_size
        self.nfeatures = nfeatures
        self.detector = self.create_detector()

    @abstractmethod
    def create_detector(self):
        """Создает детектор OpenCV."""

    def postprocess(self, descriptors):
        """Приводит дескрипторы OpenCV к вещественной матрице N x D."""
        return descriptors.astype(np.float32)

    def get_params(self):
        return {'image_size': self.image_size, 'nfeatures': self.nfeatures}

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(resize_keep_aspect(image, self.image_size), cv2.COLOR_BGR2GRAY)
        _, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None: #на изображении не нашлось ни одной точки
            return np.empty((0, self.descriptor_size), dtype=np.float32)
        return self.postprocess(descriptors)


class SiftExtractor(ClassicalExtractor):
    """SIFT: экстремумы разности гауссиан и гистограммы направлений градиента 4 x 4 x 8 = 128.
    Дескрипторы переводятся в RootSIFT: L1-нормировка и квадратный корень из каждого элемента."""
    name = 'sift'
    descriptor_size = 128

    def create_detector(self):
        return cv2.SIFT_create(nfeatures=self.nfeatures)

    def postprocess(self, descriptors):
        descriptors = descriptors.astype(np.float32)
        descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
        return np.sqrt(descriptors)


class OrbExtractor(ClassicalExtractor):
    """ORB: углы FAST на пирамиде масштабов и бинарный дескриптор rBRIEF из 256 бит.
    Биты распаковываются в вектор из 0 и 1: квадрат евклидова расстояния между такими
    векторами равен расстоянию Хэмминга, поэтому к ним применим обычный метод K-средних."""
    name = 'orb'
    descriptor_size = 256

    def create_detector(self):
        return cv2.ORB_create(nfeatures=self.nfeatures, fastThreshold=10)

    def postprocess(self, descriptors):
        return np.unpackbits(descriptors, axis=1).astype(np.float32) #разворачивает каждый из 32 байтов в 8 отдельных битов, превращая матрицу (N, 32) в матрицу (N, 256) из нулей и единиц


class CnnExtractorBase(FeatureExtractor):
    """Базовый класс для нейросетевых экстракторов. Признаки берутся с выхода слоя layer сети, обученной на ImageNet"""
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) #статистика ImageNet в порядке RGB
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, layer, input_size=288):
        import torch
        from torchvision.models.feature_extraction import create_feature_extractor

        if input_size < 32:
            raise ValueError('input_size должен быть не меньше 32')
        self.layer = layer
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.build_model().eval()
        try:
            #сеть обрезается после слоя self.layer и возвращает словарь {self.layer: выход этого слоя}
            self.model = create_feature_extractor(model, return_nodes=[self.layer]).to(self.device)
        except ValueError as e:
            raise ValueError(f'В сети {self.name} нет слоя {self.layer}') from e
        logger.info('Сеть %s, признаки берутся с выхода слоя %s, устройство %s', self.name, self.layer, self.device)

    @abstractmethod
    def build_model(self):
        """Создает модель torchvision с весами ImageNet."""

    def get_params(self):
        return {'layer': self.layer, 'input_size': self.input_size}

    def preprocess(self, image):
        """Меньшая сторона приводится к input_size, большая умножается на тот же коэффициент,
        BGR -> RGB, значения в [0, 1], нормировка (x - mean) / std, раскладка HWC -> 1 x C x H x W."""
        import torch

        h, w = image.shape[:2]
        scale = self.input_size / min(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgb = cv2.cvtColor(cv2.resize(image, size, interpolation=interpolation), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255 - self.MEAN) / self.STD
        return torch.from_numpy(tensor.transpose(2, 0, 1).copy()).unsqueeze(0)

    def get_layer_output(self, image):
        """Прямой проход сети до слоя self.layer. Возвращает выход этого слоя: карту признаков C x h x w."""
        import torch

        with torch.no_grad():
            outputs = self.model(self.preprocess(image).to(self.device))
        return outputs[self.layer][0].cpu().numpy() #[0]: в батче одно изображение

    def detect_and_compute(self, image):
        feature_map = self.get_layer_output(image)
        c, fh, fw = feature_map.shape
        descriptors = feature_map.reshape(c, fh * fw).T.astype(np.float32) #строка = вектор одной ячейки карты
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7 #L2: убираем зависимость от контраста
        return descriptors


class ResNet50Extractor(CnnExtractorBase):
    """ResNet-50, веса IMAGENET1K_V2. По умолчанию признаки с выхода слоя layer4: 2048 каналов, шаг 32 пикселя."""
    name = 'resnet50'

    def __init__(self, layer='layer4', input_size=288):
        super().__init__(layer, input_size)

    def build_model(self):
        from torchvision.models import ResNet50_Weights, resnet50
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


class EfficientNetB0Extractor(CnnExtractorBase):
    """EfficientNet-B0, веса IMAGENET1K_V1. По умолчанию признаки с выхода слоя features.8: 1280 каналов, шаг 32 пикселя."""
    name = 'efficientnet_b0'

    def __init__(self, layer='features.8', input_size=288):
        super().__init__(layer, input_size)

    def build_model(self):
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)


EXTRACTORS = {cls.name: cls for cls in (SiftExtractor, OrbExtractor, ResNet50Extractor, EfficientNetB0Extractor)}
CLASSICAL = ('sift', 'orb')
