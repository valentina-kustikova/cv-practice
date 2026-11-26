from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock

import cv2
import numpy as np

NEURAL_EXTRACTORS = ("resnet18", "resnet50", "mobilenet_v3_small", "dinov2")
LAYER_EXTRACTORS = ("resnet18", "resnet50", "dinov2")


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
            "dinov2": DinoV2Extractor,
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

    def __init__(self, device: str = "auto", pretrained: bool = True, window_size: int = 224) -> None:
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        try:
            import torch
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
        self.window_size = window_size
        self.model = self._build_model(pretrained).to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(round(window_size * 8 / 7)),
                transforms.CenterCrop(window_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @abstractmethod
    def _build_model(self, pretrained: bool):
        """Build the torchvision model used for extraction."""

    def _feature_map(self, tensor):
        return self.model(tensor)

    def detect_and_compute(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("CNN extractors expect a 3-channel BGR image")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            feature_map = self._feature_map(tensor)

        if feature_map.ndim != 4:
            raise RuntimeError(
                f"Expected a CNN feature map [N,C,H,W], received shape {tuple(feature_map.shape)}"
            )
        # [1,C,H,W] -> [H*W,C], one descriptor for each spatial location.
        descriptors = feature_map[0].permute(1, 2, 0).reshape(-1, feature_map.shape[1])
        return descriptors.detach().cpu().numpy().astype(np.float32, copy=False)


class ResNetExtractorBase(CnnExtractorBase):
    """Read a spatial activation during the original ResNet forward pass."""

    # Fixed indices for the spatial outputs common to both architectures.
    STAGES = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")
    DEFAULT_LAYER = STAGES.index("layer4")

    def __init__(
        self, device: str = "auto", pretrained: bool = True, window_size: int = 224, layer: int = DEFAULT_LAYER,
    ) -> None:
        super().__init__(device=device, pretrained=pretrained, window_size=window_size)
        nested = (name for name, _ in self.model.named_modules() if "." in name)
        self.layer_names = (*self.STAGES, *nested)
        if not isinstance(layer, int) or not 0 <= layer < len(self.layer_names):
            raise ValueError(f"layer must be an integer between 0 and {len(self.layer_names) - 1}")
        self.layer = layer
        self._target = self.model.get_submodule(self.layer_names[layer])
        self._forward_lock = Lock()

    def _feature_map(self, tensor):
        activation = None

        def capture(module, inputs, output):
            nonlocal activation
            # Later in-place ReLU and residual additions may modify this tensor.
            activation = output.clone()

        with self._forward_lock, self._target.register_forward_hook(capture):
            self.model(tensor)
        if activation is None:
            raise RuntimeError(f"ResNet did not execute layer {self.layer_names[self.layer]}")
        return activation


class ResNet18Extractor(ResNetExtractorBase):
    """Local descriptors from a selected ResNet-18 feature map."""

    def _build_model(self, pretrained: bool):
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)


class ResNet50Extractor(ResNetExtractorBase):
    """Local descriptors from a selected ResNet-50 feature map."""

    def _build_model(self, pretrained: bool):
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)


class MobileNetV3SmallExtractor(CnnExtractorBase):
    """ImageNet-pretrained MobileNetV3-Small features before global pooling."""

    def _build_model(self, pretrained: bool):
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        return base.features


class DinoV2Extractor(CnnExtractorBase):
    """Local patch descriptors from a DINOv2 ViT-B/14 transformer block."""

    layer_names = tuple(f"blocks.{i}" for i in range(12))

    def __init__(self, device: str = "auto", pretrained: bool = True, window_size: int = 224, layer: int = 11):
        if not isinstance(window_size, int) or window_size < 14 or window_size % 14:
            raise ValueError("DINOv2 --window-size must be a positive multiple of 14")
        if not isinstance(layer, int) or not 0 <= layer < len(self.layer_names):
            raise ValueError("DINOv2 --layer must be a block index from 0 to 11")
        self.layer = layer
        super().__init__(device=device, pretrained=pretrained, window_size=window_size)

    def _build_model(self, pretrained: bool):
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("DINOv2 requires timm. Install the dependencies in requirements.txt.") from exc
        return timm.create_model(
            "vit_base_patch14_dinov2.lvd142m", pretrained=pretrained, img_size=self.window_size,
            pretrained_cfg_overlay={"hf_hub_id": None},  # Load Meta's original checkpoint.
        )

    def _feature_map(self, tensor):
        return self.model.forward_intermediates(
            tensor, indices=[self.layer], norm=True, stop_early=True, intermediates_only=True,
        )[0]
