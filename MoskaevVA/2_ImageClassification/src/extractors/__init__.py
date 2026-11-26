from .base import FeatureExtractor
from .classical import ClassicalExtractor, SIFTExtractor, KAZEExtractor
try:
    from .cnn import CnnExtractorBase, ResNet18Extractor, ConvNeXtTinyExtractor
except ImportError:
    CnnExtractorBase = ResNet18Extractor = ConvNeXtTinyExtractor = None

__all__ = [
    "FeatureExtractor", "ClassicalExtractor", "SIFTExtractor", "KAZEExtractor",
    "CnnExtractorBase", "ResNet18Extractor", "ConvNeXtTinyExtractor",
]