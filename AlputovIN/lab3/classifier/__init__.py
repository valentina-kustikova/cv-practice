"""
Пакет с классификаторами для классификации изображений достопримечательностей
"""

from .base_classifier import BaseClassifier
from .bow_classifier import BOWClassifier
from .cnn_classifier import CNNClassifier

__all__ = ['BaseClassifier', 'BOWClassifier', 'CNNClassifier']

