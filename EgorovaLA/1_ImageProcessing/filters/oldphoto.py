import numpy as np

from .antique import Antique
from .base import ImageFilter
from .utils import blur, to_uint8


class OldPhoto(ImageFilter):
    """Состаренное фото."""

    name = "oldphoto"

    def __init__(self, noise=0.4, scratches=20, spots=15, creases=1, vignette=0.6, seed=None):
        self.noise = noise
        self.scratches = scratches
        self.spots = spots
        self.creases = creases
        self.vignette = vignette
        self.seed = seed

    def apply_filter(self, image):
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # мягкость, сепия, выцветание
        img = blur(image, max(h, w) / 1000)
        img = Antique().apply_filter(to_uint8(img)).astype(np.float32)
        img = 25 + 0.8 * img

        # виньетка
        r2 = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
        img *= np.clip(1 - 0.6 * self.vignette * r2, 0, 1)[..., None]

        # зерно
        img += rng.normal(0, 45 * self.noise, (h, w)).astype(np.float32)[..., None]

        # заломы
        width = max(1.0, min(h, w) / 350)
        for _ in range(self.creases):
            angle = rng.uniform(0, np.pi)
            px, py = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
            along = -(x - px) * np.sin(angle) + (y - py) * np.cos(angle)
            d = (x - px) * np.cos(angle) + (y - py) * np.sin(angle) + width * np.sin(along / (60 * width))
            band = np.abs(d) < 10 * width
            db = d[band]
            light = 0.55 * np.exp(-(db / width) ** 2)
            shadow = 0.3 * np.exp(-((db - 2.5 * width) / (1.5 * width)) ** 2)
            cracks = rng.normal(0, 22, db.size) * np.exp(-(db / (4 * width)) ** 2)
            pix = img[band] * (1 - shadow)[:, None]
            img[band] = pix + (235 - pix) * light[:, None] + cracks[:, None]

        # царапины
        for _ in range(self.scratches):
            length = rng.integers(h // 4, h + 1)
            y0 = rng.integers(0, h - length + 1)
            xs = np.clip(rng.integers(0, w) + np.cumsum(rng.choice([-1, 0, 0, 0, 1], length)), 0, w - 1)
            ys = np.arange(y0, y0 + length)
            tone = 235 if rng.random() < 0.8 else 30
            a = rng.uniform(0.3, 0.7)
            img[ys, xs] = (1 - a) * img[ys, xs] + a * tone

        # пятна
        for _ in range(self.spots):
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            dark = rng.random() < 0.3
            radius = rng.uniform(0.5, 1.2) if dark else rng.uniform(1, max(2, min(h, w) / 150))
            y1, y2 = int(max(0, cy - 2 * radius)), int(min(h, cy + 2 * radius + 1))
            x1, x2 = int(max(0, cx - 2 * radius)), int(min(w, cx + 2 * radius + 1))
            d = np.sqrt((y[y1:y2, x1:x2] - cy) ** 2 + (x[y1:y2, x1:x2] - cx) ** 2) / radius
            alpha = np.clip(1.5 - d, 0, 1)[..., None] * rng.uniform(0.3, 0.6)
            tone = 40 if dark else 225
            img[y1:y2, x1:x2] = (1 - alpha) * img[y1:y2, x1:x2] + alpha * tone

        # потёртый край
        border = np.minimum.reduce([x, y, w - 1 - x, h - 1 - y]) / min(h, w)
        edge = 0.02 + 0.01 * rng.random((h, w))
        alpha = np.clip((edge - border) * min(h, w) / 2, 0, 1)[..., None]
        img = (1 - alpha) * img + alpha * np.array([190, 215, 228], np.float32)

        return to_uint8(img)
