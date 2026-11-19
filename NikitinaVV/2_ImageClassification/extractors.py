import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_longer_side(image, size):
    """Масштабирует изображение так, чтобы большая сторона стала равна size. Возвращает изображение и масштаб."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC #при уменьшении INTER_AREA не дает муара
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)
    return resized, scale


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
        """Принять картинку, вернуть (keypoints, descriptors) в едином формате."""

    @abstractmethod
    def get_params(self):
        """вернуть словарь параметров конструктора (чтобы сохранить их в файл модели и при загрузке воссоздать точно такой же экстрактор)"""

    @property
    @abstractmethod
    def min_side(self):
        """Подсказать функции read_image, до какого размера можно смело сжимать JPEG прямо при чтении с диска"""


class ClassicalExtractor(FeatureExtractor):
    """Общая часть классических методов OpenCV: масштабирование, перевод в серый, detectAndCompute.
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

    @property
    def min_side(self):
        return self.image_size

    def detect_and_compute(self, image):
        resized, scale = resize_longer_side(image, self.image_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return [], np.empty((0, self.descriptor_size), dtype=np.float32)
        #возвращаем точки в координаты исходного изображения
        keypoints = [cv2.KeyPoint(kp.pt[0] / scale, kp.pt[1] / scale, kp.size / scale, kp.angle) for kp in keypoints]
        return keypoints, self.postprocess(descriptors)


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
    """Базовый класс для нейросетевых экстракторов. Признаки с заданного сверточного слоя сети, обученной на ImageNet"""
    default_layer = None
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) #статистика ImageNet в порядке RGB
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, layer=None, input_size=288):
        import torch
        from torchvision.models.feature_extraction import create_feature_extractor

        if input_size < 32:
            raise ValueError('input_size должен быть не меньше 32')
        self.layer = layer or self.default_layer
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.build_model().eval()
        try:
            self.model = create_feature_extractor(model, return_nodes={self.layer: 'features'}).to(self.device)
        except ValueError as e:
            raise ValueError(f'В сети {self.name} нет слоя {self.layer}') from e
        logger.info('Сеть %s, слой %s, устройство %s', self.name, self.layer, self.device)

    @abstractmethod
    def build_model(self):
        """Создает модель torchvision с весами ImageNet."""

    def get_params(self):
        return {'layer': self.layer, 'input_size': self.input_size}

    @property
    def min_side(self):
        return self.input_size

    def preprocess(self, image):
        """Меньшая сторона приводится к input_size с сохранением пропорций, BGR -> RGB,
        значения в [0, 1], нормировка (x - mean) / std, раскладка HWC -> 1 x C x H x W."""
        import torch

        h, w = image.shape[:2]
        scale = self.input_size / min(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgb = cv2.cvtColor(cv2.resize(image, size, interpolation=interpolation), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255 - self.MEAN) / self.STD
        return torch.from_numpy(tensor.transpose(2, 0, 1).copy()).unsqueeze(0)

    def detect_and_compute(self, image):
        import torch

        with torch.no_grad():
            features = self.model(self.preprocess(image).to(self.device))['features'][0].cpu().numpy()
        c, fh, fw = features.shape
        descriptors = features.reshape(c, fh * fw).T.astype(np.float32) #строка = вектор одной ячейки карты
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7 #L2: убираем зависимость от контраста

        #ключевая точка = центр ячейки карты признаков на исходном изображении
        h, w = image.shape[:2]
        cell_w, cell_h = w / fw, h / fh
        keypoints = [cv2.KeyPoint((j + 0.5) * cell_w, (i + 0.5) * cell_h, min(cell_w, cell_h))
                     for i in range(fh) for j in range(fw)]
        return keypoints, descriptors


class ResNet50Extractor(CnnExtractorBase):
    """ResNet-50, веса IMAGENET1K_V2. Слой layer4: 2048 каналов, шаг 32 пикселя."""
    name = 'resnet50'
    default_layer = 'layer4'

    def build_model(self):
        from torchvision.models import ResNet50_Weights, resnet50
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


class EfficientNetB0Extractor(CnnExtractorBase):
    """EfficientNet-B0, веса IMAGENET1K_V1. Слой features.8: 1280 каналов, шаг 32 пикселя."""
    name = 'efficientnet_b0'
    default_layer = 'features.8'

    def build_model(self):
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)


EXTRACTORS = {cls.name: cls for cls in (SiftExtractor, OrbExtractor, ResNet50Extractor, EfficientNetB0Extractor)}
CLASSICAL = ('sift', 'orb')
