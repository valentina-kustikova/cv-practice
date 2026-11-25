"""Извлекает локальные дескрипторы из RGB-изображения."""
from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import logging
import cv2
import numpy as np


class FeatureExtractor(ABC):
    """Задаёт общий интерфейс для извлечения локальных дескрипторов."""

    @abstractmethod
    def detect_and_compute(self, image):
        """Возвращает дескрипторы изображения в виде массива (число дескрипторов, размерность)."""
        pass

    @staticmethod
    def create(name, n_features=300, device='cpu', weights_dir='weights', expected_sha256=None):
        """Создаёт экстрактор с указанным именем и параметрами."""
        if name == 'sift':
            return SiftExtractor(n_features)
        if name == 'orb':
            return OrbExtractor(n_features)
        if name == 'resnet18':
            return ResNetExtractor(weights_dir, device, expected_sha256)
        if name == 'mobilenetv2':
            return MobileNetExtractor(weights_dir, device, expected_sha256)
        raise ValueError(f'Неизвестный экстрактор: {name}')


class ClassicalExtractor(FeatureExtractor):
    """Извлекает признаки из полутонового изображения с помощью OpenCV."""

    def detect_and_compute(self, image):
        """Находит ключевые точки и возвращает их дескрипторы."""
        _, descriptors = self.detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        if descriptors is None:
            return np.empty((0, self.dimension), dtype=np.float32)
        return descriptors[:self.n_features].astype(np.float32)

    @abstractmethod
    def __init__(self, n_features=300):
        """Сохраняет максимальное число дескрипторов изображения."""
        if n_features < 1:
            raise ValueError('Число признаков должно быть положительным')
        self.n_features = n_features


class SiftExtractor(ClassicalExtractor):
    """Извлекает 128-мерные дескрипторы SIFT."""
    dimension = 128

    def __init__(self, n_features=300):
        """Создаёт детектор SIFT с ограничением числа признаков."""
        super().__init__(n_features)
        self.detector = cv2.SIFT_create(nfeatures=n_features)


class OrbExtractor(ClassicalExtractor):
    """Описывает каждую ключевую точку ORB вектором из 32 байтов."""
    dimension = 32

    def __init__(self, n_features=300):
        """Создаёт детектор ORB с ограничением числа признаков."""
        super().__init__(n_features)
        self.detector = cv2.ORB_create(nfeatures=n_features)


class CnnExtractorBase(FeatureExtractor):
    """Загружает свёрточную сеть и извлекает локальные дескрипторы из её карты признаков."""

    @abstractmethod
    def architecture(self):
        """Возвращает функцию создания модели, описание весов и функцию выбора слоёв."""

    def __init__(self, weights_dir='weights', device='cpu', expected_sha256=None):
        """Загружает веса и подготавливает выбранную часть сети к извлечению признаков."""
        import torch
        self.torch = torch
        factory, weights, truncate = self.architecture()
        path = Path(weights_dir) / Path(weights.url).name
        if not path.is_file():
            raise FileNotFoundError(f'Файл весов CNN не найден: {path}. Запустите python download_weights.py '
                                    'на компьютере с интернетом и скопируйте папку weights.')
        self.weights_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and self.weights_sha256 != expected_sha256:
            raise ValueError('Веса CNN отличаются от использованных при обучении')
        model = factory(weights=None)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.model = truncate(model).eval().to(device)
        self.model.requires_grad_(False)
        self.transform = weights.transforms()
        self.device = device
        self._logged_shape = False

    def detect_and_compute(self, image):
        """Преобразует карту признаков CNN в набор локальных дескрипторов."""
        from PIL import Image
        with self.torch.inference_mode():
            tensor = self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)
            features = self.model(tensor)[0]
            if tuple(features.shape) != self.map_shape:
                raise ValueError(f'Неожиданный размер карты признаков: {tuple(features.shape)}')
            if not self._logged_shape:
                logging.info('%s: карта признаков %s, дескрипторы (%d, %d)', type(self).__name__,
                             tuple(features.shape), features.shape[1]*features.shape[2], features.shape[0])
                self._logged_shape = True
            return features.permute(1, 2, 0).reshape(-1, features.shape[0]).cpu().numpy().copy()


class ResNetExtractor(CnnExtractorBase):
    """Извлекает признаки из последнего свёрточного блока ResNet18."""

    dimension = 512
    map_shape = (512, 7, 7)

    def architecture(self):
        """Возвращает настройки ResNet18 для извлечения признаков со слоя layer4."""
        from torch import nn
        from torchvision.models import resnet18, ResNet18_Weights
        return resnet18, ResNet18_Weights.IMAGENET1K_V1, lambda m: nn.Sequential(*list(m.children())[:8])


class MobileNetExtractor(CnnExtractorBase):
    """Извлекает признаки из блока features[17] сети MobileNetV2."""

    dimension = 320
    map_shape = (320, 7, 7)

    def architecture(self):
        """Возвращает настройки MobileNetV2 для извлечения признаков из блока features[17]."""
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        return mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1, lambda m: m.features[:18]
