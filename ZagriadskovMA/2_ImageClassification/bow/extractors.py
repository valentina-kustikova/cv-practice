"""Извлечение локальных признаков: классические дескрипторы (OpenCV) и CNN (PyTorch)."""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .utils import resize_keep_aspect

logger = logging.getLogger(__name__)


class FeatureExtractor(ABC):
    """Абстрактный интерфейс извлечения локальных дескрипторов из изображения.

    Наследники регистрируются автоматически по атрибуту ``name``, после чего
    их можно создавать фабричным методом ``FeatureExtractor.create(name, ...)``.
    """

    name = None
    default_pca_dim = None  # рекомендуемая размерность PCA перед K-средних (None – без PCA)
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            FeatureExtractor._registry[cls.name] = cls

    @classmethod
    def create(cls, name, **params):
        """Фабричный метод: создает экстрактор по его имени (sift, orb, resnet50, ...)."""
        if name not in cls._registry:
            raise ValueError(f"Неизвестный метод '{name}'. Доступны: {cls.available()}")
        return cls._registry[name](**params)

    @classmethod
    def available(cls):
        return sorted(cls._registry)

    @abstractmethod
    def detect_and_compute(self, image):
        """Вычисляет локальные дескрипторы изображения.

        :param image: BGR-изображение (np.ndarray, uint8).
        :return: (keypoints, descriptors), где keypoints – список cv2.KeyPoint
                 в координатах исходного изображения, descriptors – np.ndarray
                 float32 размера N x D (N может быть 0).
        """

    @property
    @abstractmethod
    def descriptor_size(self):
        """Размерность D одного дескриптора после преобразования к float32."""

    def get_params(self):
        """Параметры конструктора; сохраняются в файл модели, чтобы восстановить экстрактор."""
        return dict(self._params)


class ClassicalExtractor(FeatureExtractor):
    """Базовый класс классических детекторов/дескрипторов OpenCV.

    Изображения Intel Image Classification имеют размер 150x150, на них детекторы
    находят мало точек, поэтому изображение предварительно увеличивается до
    ``image_size`` по большей стороне и переводится в оттенки серого.
    """

    def __init__(self, image_size=300, nfeatures=1000):
        self._params = {"image_size": image_size, "nfeatures": nfeatures}
        self.image_size = image_size
        self.detector = self._create_detector(nfeatures)

    @abstractmethod
    def _create_detector(self, nfeatures):
        """Создает объект детектора OpenCV (cv2.SIFT, cv2.ORB, ...)."""

    def _postprocess(self, descriptors):
        """Приводит дескрипторы OpenCV к float32-матрице N x D."""
        return descriptors.astype(np.float32)

    def detect_and_compute(self, image):
        h, w = image.shape[:2]
        resized = resize_keep_aspect(image, self.image_size, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return [], np.empty((0, self.descriptor_size), dtype=np.float32)

        # Координаты ключевых точек возвращаем в масштабе исходного изображения
        scale = max(h, w) / max(resized.shape[:2])
        keypoints = [cv2.KeyPoint(kp.pt[0] * scale, kp.pt[1] * scale, kp.size * scale, kp.angle)
                     for kp in keypoints]
        return keypoints, self._postprocess(descriptors)


class SiftExtractor(ClassicalExtractor):
    """SIFT (Lowe, 2004): 128-мерные гистограммы ориентаций градиентов.

    При ``root_sift=True`` применяется RootSIFT (Arandjelovic & Zisserman, 2012):
    L1-нормализация и поэлементный квадратный корень, что соответствует ядру
    Хеллингера и обычно повышает качество BoW.
    """

    name = "sift"

    def __init__(self, image_size=300, nfeatures=1000, root_sift=True):
        super().__init__(image_size=image_size, nfeatures=nfeatures)
        self._params["root_sift"] = root_sift
        self.root_sift = root_sift

    def _create_detector(self, nfeatures):
        return cv2.SIFT_create(nfeatures=nfeatures)

    @property
    def descriptor_size(self):
        return 128

    def _postprocess(self, descriptors):
        descriptors = descriptors.astype(np.float32)
        if self.root_sift:
            descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
            descriptors = np.sqrt(descriptors)
        return descriptors


class OrbExtractor(ClassicalExtractor):
    """ORB (Rublee et al., 2011): детектор FAST + повернутый бинарный дескриптор BRIEF.

    Дескриптор ORB – 32 байта (256 бит). Биты распаковываются в вектор из 0/1
    длины 256: квадрат евклидова расстояния между такими векторами равен
    расстоянию Хэмминга, поэтому к ним корректно применять K-средних.
    """

    name = "orb"

    def __init__(self, image_size=300, nfeatures=1000, fast_threshold=10):
        self.fast_threshold = fast_threshold
        super().__init__(image_size=image_size, nfeatures=nfeatures)
        self._params["fast_threshold"] = fast_threshold

    def _create_detector(self, nfeatures):
        # Пониженный порог FAST нужен для однородных сцен (море, ледник), где точек мало
        return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=self.fast_threshold)

    @property
    def descriptor_size(self):
        return 256

    def _postprocess(self, descriptors):
        return np.unpackbits(descriptors, axis=1).astype(np.float32)


class CnnExtractorBase(FeatureExtractor):
    """Базовый класс извлечения локальных признаков со сверточного слоя CNN.

    Выход слоя размера C x H x W рассматривается как H*W локальных дескрипторов
    размерности C: каждый описывает свою область изображения (рецептивное поле).

    Предобработка (совпадает с обучением моделей на ImageNet):
      1. resize до input_size x input_size (cv2.resize, бикубическая интерполяция);
      2. cv2.dnn.blobFromImage: BGR -> RGB, деление на 255, HWC -> NCHW;
      3. нормализация по каналам: (x - mean) / std со статистикой ImageNet.
    Постобработка: тензор 1 x C x H x W -> матрица (H*W) x C, L2-нормализация строк.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    default_layer = None
    default_pca_dim = 128

    def __init__(self, layer=None, input_size=320, device="auto", l2_normalize=True):
        from torchvision.models.feature_extraction import create_feature_extractor

        self.layer = layer or self.default_layer
        self.input_size = input_size
        self.l2_normalize = l2_normalize
        self.device = self._resolve_device(device)
        self._params = {"layer": self.layer, "input_size": input_size,
                        "device": device, "l2_normalize": l2_normalize}

        model = self._build_model().eval()
        self.model = create_feature_extractor(model, return_nodes={self.layer: "features"}).to(self.device)
        self._descriptor_size = None

    @staticmethod
    def _resolve_device(device):
        """Выбирает устройство PyTorch: auto – CUDA при наличии, иначе CPU.

        Если CUDA запрошена, но недоступна (нет видеокарты или установлена CPU-сборка
        PyTorch), работа продолжается на CPU с предупреждением. Поэтому модель,
        обученную на GPU, можно применять на компьютере без видеокарты.
        """
        import torch

        cuda_ok = torch.cuda.is_available()
        if device == "auto":
            device = "cuda" if cuda_ok else "cpu"
        elif device.startswith("cuda") and not cuda_ok:
            reason = ("установлена CPU-сборка PyTorch" if torch.version.cuda is None
                      else "видеокарта с поддержкой CUDA не найдена")
            logger.warning("CUDA недоступна (%s), используется CPU", reason)
            device = "cpu"
        logger.info("Устройство PyTorch: %s", device)
        return torch.device(device)

    @abstractmethod
    def _build_model(self):
        """Создает модель torchvision с весами ImageNet."""

    def preprocess(self, image):
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0 / 255, swapRB=True, crop=False)
        return (blob - self.MEAN) / self.STD

    def detect_and_compute(self, image):
        import torch

        blob = self.preprocess(image)
        with torch.no_grad():
            features = self.model(torch.from_numpy(blob).to(self.device))["features"]
        features = features[0].cpu().numpy()                           # C x H x W
        c, fh, fw = features.shape
        descriptors = features.reshape(c, fh * fw).T.astype(np.float32)  # (H*W) x C
        if self.l2_normalize:
            descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7
        self._descriptor_size = c

        # "Ключевые точки" – центры ячеек карты признаков в координатах исходного изображения
        h, w = image.shape[:2]
        cell_w, cell_h = w / fw, h / fh
        keypoints = [cv2.KeyPoint((j + 0.5) * cell_w, (i + 0.5) * cell_h, min(cell_w, cell_h))
                     for i in range(fh) for j in range(fw)]
        return keypoints, descriptors

    @property
    def descriptor_size(self):
        if self._descriptor_size is None:
            self.detect_and_compute(np.zeros((self.input_size, self.input_size, 3), np.uint8))
        return self._descriptor_size


class ResNet50Extractor(CnnExtractorBase):
    """ResNet-50 (He et al., 2016), веса IMAGENET1K_V2.

    Слой по умолчанию – layer4 (2048 каналов, шаг 32): при input_size=320 дает
    карту 10x10, то есть 100 дескрипторов на изображение.
    """

    name = "resnet50"
    default_layer = "layer4"

    def _build_model(self):
        from torchvision.models import resnet50, ResNet50_Weights
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


class EfficientNetB0Extractor(CnnExtractorBase):
    """EfficientNet-B0 (Tan & Le, 2019), веса IMAGENET1K_V1.

    Слой по умолчанию – features.8 (последняя свертка 1x1, 1280 каналов, шаг 32):
    при input_size=320 дает карту 10x10.
    """

    name = "efficientnet_b0"
    default_layer = "features.8"

    def _build_model(self):
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
