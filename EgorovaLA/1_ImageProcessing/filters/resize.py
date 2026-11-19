import numpy as np

from .base import ImageFilter


class Resize(ImageFilter):
    name = "resize"

    def __init__(self, width=None, height=None):
        self.width = width
        self.height = height

    def apply_filter(self, image):
        h, w = image.shape[:2]

        new_w = int(self.width) if self.width else w // 2
        new_h = int(self.height) if self.height else h // 2

        # метод ближайшего соседа
        rows = np.arange(new_h) * h // new_h
        cols = np.arange(new_w) * w // new_w
        return image[rows][:, cols]
