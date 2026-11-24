
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
import torchvision


class FeatureExtractor(ABC):
    descriptor_size = 0

    @abstractmethod
    def detect_and_compute(self, image):
        """Вычисляет дескрипторы изображения."""

    def empty_descriptors(self):
        return np.zeros((0, self.descriptor_size), dtype=np.float32)

    @staticmethod
    def available_names():
        return ["sift", "orb", "alexnet", "resnet101"]

    @staticmethod
    def create(name, device="cpu"):
        """
            name: sift, orb, alexnet или resnet101.
            device: cpu или cuda.
        """
        name = name.lower()
        if name == "sift":
            return SiftExtractor()
        if name == "orb":
            return OrbExtractor()
        if name == "alexnet":
            return AlexNetExtractor(device=device)
        if name == "resnet101":
            return ResNet101Extractor(device=device)
        raise ValueError(f"Неизвестный метод извлечения признаков: {name}. "
                         f"Допустимые значения: {', '.join(FeatureExtractor.available_names())}")


# ---------------------------------------------------------------------------
# Классические методы
# ---------------------------------------------------------------------------

class ClassicalExtractor(FeatureExtractor):
    def __init__(self, max_side=800):
        self.max_side = max_side

    def preprocess(self, image):
        # Перевод в оттенки серого и уменьшение размера
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        longest_side = max(height, width)
        if longest_side > self.max_side:
            scale = self.max_side / longest_side
            new_size = (int(round(width * scale)), int(round(height * scale)))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        return gray

    @abstractmethod
    def compute_raw(self, gray):
        pass

    def convert_descriptors(self, raw_descriptors):
        return raw_descriptors.astype(np.float32)

    def detect_and_compute(self, image):
        gray = self.preprocess(image)
        raw_descriptors = self.compute_raw(gray)
        if raw_descriptors is None or len(raw_descriptors) == 0:
            return self.empty_descriptors()
        return self.convert_descriptors(raw_descriptors)


class SiftExtractor(ClassicalExtractor):
    descriptor_size = 128 # по 8 бинов для 16 блоков 4 x 4

    def __init__(self, max_side=800, max_keypoints=1000):
        super().__init__(max_side)
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints)

    def compute_raw(self, gray):
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors


class OrbExtractor(ClassicalExtractor):
    descriptor_size = 256 # бинарный дескриптор повернутого BRIEF 

    def __init__(self, max_side=800, max_keypoints=1000):
        super().__init__(max_side)
        self.orb = cv2.ORB_create(nfeatures=max_keypoints)

    def compute_raw(self, gray):
        _, descriptors = self.orb.detectAndCompute(gray, None)
        return descriptors

    def convert_descriptors(self, raw_descriptors):
        bits = np.unpackbits(raw_descriptors, axis=1) # распаковка битового поля для k-means
        return bits.astype(np.float32)


# ---------------------------------------------------------------------------
# Нейросетевые методы
# ---------------------------------------------------------------------------

class CnnExtractorBase(FeatureExtractor):
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, device="cpu", input_size=224):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна, используйте --device cpu")
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = self.build_model()
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def build_model(self):
        pass

    def preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - self.IMAGENET_MEAN) / self.IMAGENET_STD
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(self, feature_map):
        # feature_map: тензор 1 x C x H x W
        feature_map = feature_map[0]
        channels = feature_map.shape[0]
        descriptors = feature_map.reshape(channels, -1).transpose(0, 1)  # (H*W) x C
        descriptors = descriptors.cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # защита от деления на ноль для нулевых векторов
        return descriptors / norms

    def detect_and_compute(self, image):
        tensor = self.preprocess(image)
        with torch.no_grad():
            feature_map = self.model(tensor)
        return self.postprocess(feature_map)


class AlexNetExtractor(CnnExtractorBase):
    descriptor_size = 256

    def build_model(self):
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        alexnet = torchvision.models.alexnet(weights=weights)
        # alexnet.features: [conv1, relu, pool, conv2, relu, pool,
        #                    conv3, relu, conv4, relu, conv5, relu, pool]
        return alexnet.features[:12]


class ResNet101Extractor(CnnExtractorBase):
    descriptor_size = 1024

    def build_model(self):
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        resnet = torchvision.models.resnet101(weights=weights)
        return torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
