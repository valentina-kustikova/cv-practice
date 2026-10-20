import numpy as np

from filters.base import ImageFilter
from filters.color import Antique, RGB2GrayScale


class Matte(ImageFilter):
    def apply_filter(self, img, **kwargs):
        radius = kwargs.get("matte_radius", 0.8)
        softness = kwargs.get("matte_softness", 0.2)
        if radius <= 0:
            raise ValueError("matte_radius must be positive")
        if softness < 0:
            raise ValueError("matte_softness must be non-negative")

        h, w = img.shape[:2]
        y_c, x_c = h / 2, w / 2
        y, x = np.ogrid[:h, :w]

        d = (x - x_c) ** 2 / (x_c**2) + (y - y_c) ** 2 / (y_c**2)

        if softness == 0:
            weight = (d > radius).astype(np.float32)
        else:
            weight = np.clip((d - radius) / softness, 0, 1).astype(np.float32)
        weight = weight[:, :, None]

        res = img.astype(np.float32) * (1 - weight) + 255 * weight
        return np.clip(res, 0, 255).astype(np.uint8)


class AgedPhoto(ImageFilter):
    def apply_filter(self, img, **kwargs):
        noise = kwargs.get("noise", 0.05)
        scratches = kwargs.get("scratches", 7)
        if not 0 <= noise <= 0.5:
            raise ValueError("noise must be in [0, 0.5]")
        if scratches < 0:
            raise ValueError("scratches must be non-negative")

        res = Antique().apply_filter(img).astype(np.float32)
        h, w = res.shape[:2]

        #noise_mask = np.random.rand(h, w)
        #res[noise_mask < noise] = [0, 0, 0]
        #res[noise_mask > 1 - noise] = [255, 255, 255]
        res += np.random.normal(0, noise * 255, (h, w, 1))

        for _ in range(scratches):
            x_line = np.random.randint(0, w)
            #res[:, x_line : x_line + 1] = [200, 200, 200]
            res[:, x_line] = res[:, x_line] * 0.85 + 255 * 0.15

        return np.clip(res, 0, 255).astype(np.uint8)


class NeonEffect(ImageFilter):
    def apply_filter(self, img, **kwargs):
        gain = kwargs.get("neon_gain", 9.0)
        if gain <= 0:
            raise ValueError("neon_gain must be positive")

        gray = RGB2GrayScale().apply_filter(img)[:, :, 0].astype(np.float32)

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)

        gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        gy[:-1, :] = gray[1:, :] - gray[:-1, :]

        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = np.clip(magnitude * gain, 0, 255).astype(np.uint8)

        #res = np.zeros_like(img)
        res = img.astype(np.float32)
        res[:, :, 0] += magnitude
        res[:, :, 1] += magnitude
        return np.clip(res, 0, 255).astype(np.uint8)
