from .base import ImageFilter
from .utils import to_gray, to_uint8


class GrayScale(ImageFilter):
    """Оттенки серого."""

    name = "grayscale"

    def apply_filter(self, image):
        return to_uint8(to_gray(image))
