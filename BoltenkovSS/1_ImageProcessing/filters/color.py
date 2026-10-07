import numpy as np

from filters.base import ImageFilter


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, img, **kwargs):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = np.round(0.114 * b + 0.587 * g + 0.299 * r)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


class Antique(ImageFilter):
    def apply_filter(self, img, **kwargs):
        sepia_matrix = np.array(
            [
                [0.131, 0.534, 0.272],
                [0.168, 0.686, 0.349],
                [0.189, 0.769, 0.393],
            ]
        )
        res = img.astype(np.float32).dot(sepia_matrix.T)
        return np.clip(res, 0, 255).astype(np.uint8)


class FadeColor(ImageFilter):
    def apply_filter(self, img, **kwargs):
        alpha = kwargs.get("alpha", 0.6)
        level = kwargs.get("fade_level", 150)
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not 0 <= level <= 255:
            raise ValueError("fade_level must be in [0, 255]")

        res = img.astype(np.float32) * alpha + level * (1 - alpha)
        return np.clip(res, 0, 255).astype(np.uint8)


class Infrared(ImageFilter):
    def apply_filter(self, img, **kwargs):
        gain = kwargs.get("ir_gain", 1.3)
        if gain <= 0:
            raise ValueError("ir_gain must be positive")

        b, g, r = (img[:, :, c].astype(np.float32) for c in range(3))
        new_r = np.clip(g * gain, 0, 255)
        new_g = r
        new_b = b
        return np.stack([new_b, new_g, new_r], axis=-1).astype(np.uint8)
