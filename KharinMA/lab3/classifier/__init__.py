"""
Пакет с классификаторами для классификации изображений достопримечательностей
"""

from .base_classifier import BaseClassifier
from .bow_classifier import BOWClassifier
from .efficientnet_classifier import EfficientNetClassifier

__all__ = ['BaseClassifier', 'BOWClassifier', 'EfficientNetClassifier']
