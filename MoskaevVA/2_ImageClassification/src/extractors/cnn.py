from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, ConvNeXt_Tiny_Weights

from .base import FeatureExtractor

class CnnExtractorBase(FeatureExtractor):
    """Абстрактный CNN-экстрактор; признаки снимаются с выбранного слоя через hook."""
    def __init__(self, device=None, image_size=224, layer=None, pretrained=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = int(image_size)
        self.layer_name = layer
        self.pretrained = bool(pretrained)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.model = None
        self._captured = None
        self._hook = None

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def _attach_hook(self):
        modules = dict(self.model.named_modules())
        if self.layer_name not in modules:
            candidates = ", ".join(list(modules.keys())[-30:])
            raise ValueError(f"Слой {self.layer_name!r} не найден. Последние доступные имена: {candidates}")
        def capture(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = next((v for v in output if torch.is_tensor(v)), None)
            if isinstance(output, dict):
                output = next((v for v in output.values() if torch.is_tensor(v)), None)
            if not torch.is_tensor(output):
                raise TypeError(f"Слой {self.layer_name} не вернул Tensor")
            self._captured = output
        self._hook = modules[self.layer_name].register_forward_hook(capture)

    def _tensor(self, image):
        import cv2
        if image is None:
            raise ValueError("Передано пустое изображение")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)

    def detect_and_compute(self, image):
        if self.model is None or self._hook is None:
            raise RuntimeError("CNN-модель не инициализирована")
        self._captured = None
        with torch.inference_mode():
            self.model(self._tensor(image))
            fmap = self._captured
            if fmap is None:
                raise RuntimeError(f"Hook слоя {self.layer_name!r} не был вызван")
            if fmap.ndim == 4:
                # Все пространственные позиции: (H*W, C), без уменьшения карты.
                descriptors = fmap[0].permute(1, 2, 0).reshape(-1, fmap.shape[1])
            elif fmap.ndim == 3:
                descriptors = fmap[0]
            elif fmap.ndim == 2:
                descriptors = fmap
            else:
                raise ValueError(f"Неподдерживаемая форма карты признаков: {tuple(fmap.shape)}")
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors.detach().cpu().numpy().astype(np.float32)

    def get_config(self):
        return {"name": self.name, "device": str(self.device), "image_size": self.image_size,
                "layer": self.layer_name, "pretrained": self.pretrained}

class ResNet18Extractor(CnnExtractorBase):
    name = "resnet18"
    def __init__(self, device=None, image_size=224, layer="layer4", pretrained=True):
        super().__init__(device, image_size, layer or "layer4", pretrained)
        self.model = self._build_model().to(self.device).eval()
        self._attach_hook()
    def _build_model(self):
        return resnet18(weights=ResNet18_Weights.DEFAULT if self.pretrained else None)

class ConvNeXtTinyExtractor(CnnExtractorBase):
    name = "convnext_tiny"

    def __init__(self, device=None, image_size=224, layer="features.7", pretrained=True):
        super().__init__(device, image_size, layer or "features.7", pretrained)
        self.model = self._build_model().to(self.device).eval()
        self._attach_hook()

    def _build_model(self):
        return convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if self.pretrained else None)
