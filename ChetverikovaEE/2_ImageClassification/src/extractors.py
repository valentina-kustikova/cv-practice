from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import torch
import cv2
import numpy as np

from .utils import to_grayscale

logger = logging.getLogger(__name__)

_EXTRACTOR_REGISTRY: Dict[str, Type["FeatureExtractor"]] = {}


def _register_extractor(cls: Type["FeatureExtractor"]) -> Type["FeatureExtractor"]:
    """Декоратор: добавляет класс-наследник в реестр фабрики по его атрибуту name."""
    _EXTRACTOR_REGISTRY[cls.name] = cls
    return cls


class FeatureExtractor(ABC):
    """Абстрактный извлекатель признаков из изображения.

    Любой наследник возвращает матрицу дескрипторов формы (N, D): для классических
    методов N — число найденных ключевых точек, для нейросетевых — число «позиций»
    карты признаков выбранного слоя.
    """

    name: str = "feature_extractor"  # уникальное имя для фабричного метода
    descriptor_dim: int = -1         # размерность одного дескриптора D

    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        """Вычислить дескрипторы изображения (BGR, np.ndarray).

        Возвращает матрицу (N, D) типа np.float32; при отсутствии ключевых точек —
        пустую матрицу формы (0, D), не исключение.
        """

    def get_params(self) -> Dict[str, Any]:
        """Параметры конструктора — сохраняются в файл модели вместе с именем экстрактора.

        При загрузке модели (BagOfWords.load_model) экстрактор воссоздаётся
        через FeatureExtractor.create(name, **params), поэтому параметры
        тестирования и предсказания гарантированно совпадают с обучением.
        """
        return {}

    @classmethod
    def create(cls, name: str, **params: Any) -> "FeatureExtractor":
        """Фабричный метод: по имени возвращает экземпляр нужного класса-наследника.

        :param name: имя экстрактора ('sift', 'orb', 'resnet18', 'mobilenetv2')
        :param params: параметры конструктора соответствующего класса
        :raises ValueError: если имя не зарегистрировано
        """
        if name not in _EXTRACTOR_REGISTRY:
            available = ", ".join(sorted(_EXTRACTOR_REGISTRY))
            raise ValueError(f"Неизвестный извлекатель признаков: '{name}'. Доступно: {available}")
        instance = _EXTRACTOR_REGISTRY[name](**params)
        logger.info(
            "Создан извлекатель признаков '%s' (дескриптор %d-мерный, параметры: %s)",
            name, instance.descriptor_dim, params,
        )
        return instance


class ClassicalExtractor(FeatureExtractor):
    """Абстрактный базовый класс классических методов OpenCV (SIFT, ORB, ...).

    Классические детекторы работают с полутоновым изображением, поэтому вход
    переводится в градации серого.
    """

    def __init__(self, min_side: int = 0) -> None:
        """
        :param min_side: если меньшая сторона изображения меньше этого значения,
            изображение пропорционально увеличивается (0 — не увеличивать)
        """
        self.min_side = int(min_side)
        self._detector = self._create_detector()

    @abstractmethod
    def _create_detector(self) -> Any:
        """Создать детектор OpenCV — объект с методом detectAndCompute."""

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Перевести изображение в серое и при необходимости увеличить."""
        gray = to_grayscale(image)
        if self.min_side > 0:
            height, width = gray.shape[:2]
            short_side = min(height, width)
            if short_side < self.min_side:
                scale = self.min_side / short_side
                new_size = (int(round(width * scale)), int(round(height * scale)))
                gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_LINEAR)
                logger.debug("Изображение увеличено до %dx%d для поиска ключевых точек", *new_size)
        return gray

    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        """Найти ключевые точки и вычислить дескрипторы (шаблонный метод для всех классических экстракторов)."""
        gray = self._prepare_image(image)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            logger.warning(
                "Ключевые точки не найдены (изображение %dx%d) — возвращена пустая матрица дескрипторов",
                image.shape[1], image.shape[0],
            )
            return np.empty((0, self.descriptor_dim), dtype=np.float32)
        return self._postprocess_descriptors(descriptors)

    def _postprocess_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Постобработка дескрипторов (по умолчанию — приведение к float32)."""
        return np.asarray(descriptors, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"min_side": self.min_side}


@_register_extractor
class SiftExtractor(ClassicalExtractor):
    """SIFT (Scale-Invariant Feature Transform) — cv2.SIFT_create.

    Дескриптор: 128-мерный вектор (гистограммы градиентов 4x4 ячеек по 8 направлений)
    вокруг ключевой точки. Инвариантен к масштабу и повороту, частично — к изменению
    освещённости. Ключевые точки ищутся в пирамиде разностей гауссианов (DoG).
    """

    name = "sift"
    descriptor_dim = 128

    def __init__(self, nfeatures: int = 2000, min_side: int = 0) -> None:
        """
        :param nfeatures: максимальное число ключевых точек на изображение
        :param min_side: см. ClassicalExtractor
        """
        self.nfeatures = int(nfeatures)
        super().__init__(min_side=min_side)

    def _create_detector(self) -> Any:
        return cv2.SIFT_create(nfeatures=self.nfeatures)

    def get_params(self) -> Dict[str, Any]:
        return {"nfeatures": self.nfeatures, "min_side": self.min_side}


@_register_extractor
class OrbExtractor(ClassicalExtractor):
    """ORB (Oriented FAST + Rotated BRIEF) — cv2.ORB_create.

    Ключевые точки находятся детектором FAST, дескриптор — бинарная строка BRIEF
    из 256 бит (хранится как 32 байта uint8).
    """

    name = "orb"
    descriptor_dim = 32

    def __init__(self, nfeatures: int = 2000, min_side: int = 0) -> None:
        """
        :param nfeatures: максимальное число ключевых точек на изображение
        :param min_side: см. ClassicalExtractor
        """
        self.nfeatures = int(nfeatures)
        super().__init__(min_side=min_side)

    def _create_detector(self) -> Any:
        return cv2.ORB_create(nfeatures=self.nfeatures)

    def _postprocess_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        # ORB возвращает uint8 (бинарные BRIEF-строки) — приводим к float32 для K-средних
        return np.asarray(descriptors, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"nfeatures": self.nfeatures, "min_side": self.min_side}



def resolve_device(device: str = "auto"):
    """Преобразовать строку в torch.device; 'auto' выбирает CUDA при её наличии."""
    if torch is None:
        raise ImportError("Не установлен PyTorch. Установите: pip install torch torchvision")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA недоступна — нейросеть будет работать на CPU")
        return torch.device("cpu")
    return dev


class CnnExtractorBase(FeatureExtractor):
    """Абстрактный базовый класс нейросетевых извлекателей признаков (PyTorch).

    Идея: выход промежуточного свёрточного слоя имеет форму (C, H, W). Каждый из
    H*W векторов длины C описывает локальную область исходного изображения
    (рецептивное поле), поэтому карта признаков трактуется как H*W плотных
    локальных дескрипторов размерности C. Далее они поступают в общий
    BoW-конвейер: кластеризация K-средними → гистограмма визуальных слов.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model, feature_module, input_size: int = 224,
                 device: str = "auto", normalize: bool = True) -> None:
        """
        :param model: модель torchvision (nn.Module)
        :param feature_module: слой модели, с которого снимаются признаки
        :param input_size: размер входного изображения (по умолчанию 224)
        :param device: 'auto' | 'cpu' | 'cuda'
        :param normalize: выполнять ли L2-нормировку каждого дескриптора
        """
        if torch is None:
            raise ImportError("Для нейросетевых извлекателей установите PyTorch: pip install torch torchvision")
        self.device_str = str(device)
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.feature_module = feature_module
        self.input_size = int(input_size)
        self.normalize = bool(normalize)
        self._last_feature_map = None

        feature_module.register_forward_hook(self._save_feature_map_hook)
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        self._mean = mean.to(self.device)
        self._std = std.to(self.device)

    def _save_feature_map_hook(self, module, inputs, output) -> None:
        """Forward-hook: запоминает тензор выхода целевого слоя."""
        self._last_feature_map = output.detach()

    def _preprocess(self, image: np.ndarray):
        """Предобработка изображения -> тензор (1, 3, input_size, input_size)."""
        if image.ndim == 2:  # серое изображение -> трёхканальное
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W), RGB, [0, 1]
        x = (x - self._mean) / self._std  # нормализация ImageNet
        return x.unsqueeze(0)

    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        """Извлечь плотные дескрипторы с заданного слоя сети: (H*W, C), float32."""
        x = self._preprocess(image).to(self.device)
        self._last_feature_map = None
        with torch.no_grad():
            self.model(x)
        if self._last_feature_map is None:
            raise RuntimeError("Forward-hook не сработал: проверьте переданный слой feature_module")
        feature_map = self._last_feature_map  # (1, C, H, W)
        channels = feature_map.shape[1]
        descriptors = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, channels)  # (H*W, C)
        descriptors = descriptors.cpu().numpy().astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            descriptors = descriptors / norms
        return descriptors

    def get_params(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "device": self.device_str,
            "normalize": self.normalize,
            "pretrained": self.pretrained,
        }


@_register_extractor
class ResNet18Extractor(CnnExtractorBase):
    """Признаки с последнего residual-блока ResNet-18 (слоя model.layer4).

    При входе 224x224 выход слоя имеет форму (512, 7, 7), т.е. 49 дескрипторов
    по 512 элементов. 
    """

    name = "resnet18"
    descriptor_dim = 512

    def __init__(self, input_size: int = 224, pretrained: bool = True,
                 device: str = "auto", normalize: bool = True) -> None:
        """
        :param input_size: размер входного изображения (224 по умолчанию)
        :param pretrained: загружать ли предобученные ImageNet-веса (нужен интернет при первом запуске)
        :param device: 'auto' | 'cpu' | 'cuda'
        :param normalize: L2-нормировка дескрипторов
        """
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("Установите torchvision: pip install torchvision") from exc
        self.pretrained = bool(pretrained)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = models.resnet18(weights=weights)
        super().__init__(model, model.layer4, input_size=input_size, device=device, normalize=normalize)


@_register_extractor
class MobileNetV2Extractor(CnnExtractorBase):
    """Признаки с последнего свёрточного блока MobileNetV2 (model.features[-1]).

    При входе 224x224 выход имеет форму (1280, 7, 7), т.е. 49 дескрипторов по
    1280 элементов. 
    """

    name = "mobilenetv2"
    descriptor_dim = 1280

    def __init__(self, input_size: int = 224, pretrained: bool = True,
                 device: str = "auto", normalize: bool = True) -> None:
        """
        :param input_size: размер входного изображения (224 по умолчанию)
        :param pretrained: загружать ли предобученные ImageNet-веса
        :param device: 'auto' | 'cpu' | 'cuda'
        :param normalize: L2-нормировка дескрипторов
        """
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError("Установите torchvision: pip install torchvision") from exc
        self.pretrained = bool(pretrained)
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = models.mobilenet_v2(weights=weights)
        super().__init__(model, model.features[-1], input_size=input_size, device=device, normalize=normalize)
