"""
Image filters library for image processing using OpenCV and PyQt
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .filters import ImageFilter
from .processors import OverlayProcessor, FilterProcessor, ColorConverter

__all__ = ['ImageFilter', 'OverlayProcessor', 'FilterProcessor', 'ColorConverter']

