from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class FeatureExtractor(ABC):
    """Common interface for local visual descriptor extraction."""

    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        """Return a matrix of descriptors with shape ``(n_descriptors, dimension)``."""

    @classmethod
    def create(cls, name: str, **kwargs) -> "FeatureExtractor":
        """Factory that selects an extractor by its CLI name."""
        extractors = {
            "sift": SIFTExtractor,
            "kaze": KAZEExtractor,
            "resnet18": ResNet18Extractor,
            "resnet50": ResNet50Extractor,
            "mobilenet_v3_small": MobileNetV3SmallExtractor,
        }
        normalized_name = name.lower()
        if normalized_name not in extractors:
            available = ", ".join(extractors)
            raise ValueError(f"Unknown extractor '{name}'. Available extractors: {available}")
        return extractors[normalized_name](**kwargs)


class ClassicalExtractor(FeatureExtractor, ABC):
    """Base class for classical OpenCV local-feature detectors."""

    def __init__(self, detector) -> None:
        self.detector = detector

    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        _keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            return np.empty((0, self.descriptor_size), dtype=np.float32)
        return np.asarray(descriptors, dtype=np.float32)

    @property
    @abstractmethod
    def descriptor_size(self) -> int:
        """Descriptor dimensionality used for empty-descriptor images."""


class SIFTExtractor(ClassicalExtractor):
    """Scale-Invariant Feature Transform (SIFT), 128-dimensional descriptors."""

    def __init__(self, nfeatures: int = 0) -> None:
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "This OpenCV build does not provide SIFT. Install a recent opencv-python package."
            )
        super().__init__(cv2.SIFT_create(nfeatures=nfeatures))

    @property
    def descriptor_size(self) -> int:
        return 128


class KAZEExtractor(ClassicalExtractor):
    """KAZE detector with floating-point descriptors (64 dimensions by default)."""

    def __init__(self, threshold: float = 0.001) -> None:
        super().__init__(cv2.KAZE_create(threshold=threshold))

    @property
    def descriptor_size(self) -> int:
        return 64


class CnnExtractorBase(FeatureExtractor, ABC):
    """Base class for CNN feature maps used as sets of local descriptors."""

    def __init__(self, device: str = "auto", pretrained: bool = True) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
        except Exception as exc:  
            raise RuntimeError(
                "CNN extractors require compatible PyTorch, torchvision and Pillow installations. "
                "Reinstall dependencies from requirements.txt."
            ) from exc

        self.torch = torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but it is not available in this PyTorch installation.")
            self.device = torch.device(device)
        self.pretrained = pretrained
        self.model = self._build_model(pretrained).to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self._pil_image_class = Image

    @abstractmethod
    def _build_model(self, pretrained: bool):
        """Build a model that returns a convolutional feature map [1,C,H,W]."""

    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CNN extractors expect a 3-channel BGR image")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            feature_map = self.model(tensor)

        if feature_map.ndim != 4:
            raise RuntimeError(
                f"Expected a CNN feature map [N,C,H,W], received shape {tuple(feature_map.shape)}"
            )
        # [1,C,H,W] -> [H*W,C], one descriptor for each spatial location.
        descriptors = feature_map[0].permute(1, 2, 0).reshape(-1, feature_map.shape[1])
        return descriptors.detach().cpu().numpy().astype(np.float32, copy=False)


class ResNet18Extractor(CnnExtractorBase):
    """ImageNet-pretrained ResNet-18 features from the final convolutional block."""

    def _build_model(self, pretrained: bool):
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        torch = self.torch
        return torch.nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )


class ResNet50Extractor(CnnExtractorBase):
    """ImageNet-pretrained ResNet-50 features from the final convolutional block."""

    def _build_model(self, pretrained: bool):
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        torch = self.torch
        return torch.nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )


class MobileNetV3SmallExtractor(CnnExtractorBase):
    """ImageNet-pretrained MobileNetV3-Small features before global pooling."""

    def _build_model(self, pretrained: bool):
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        return base.features
