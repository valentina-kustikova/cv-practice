import os
import numpy as np
from main import read_image

from .base import ImageFilter


TEXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "textures")

import logging

def _load_texture(name: str, size_hw):
    path = os.path.join(TEXTURE_DIR, name)
    
    if not os.path.isfile(path):
        logging.warning("Файл текстуры %s не найден, пропускаем этот слой.", name)
        return None
    
    img_rgb = read_image(path)
    
    arr = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]) / 255.0
    arr = arr.astype(np.float32)
    
    sh, sw = arr.shape[:2]
    h, w = size_hw
    
    y = (np.arange(h) * (sh / h)).astype(np.int32)
    x = (np.arange(w) * (sw / w)).astype(np.int32)
    y = np.clip(y, 0, sh - 1)
    x = np.clip(x, 0, sw - 1)
    
    return arr[y[:, None], x]


class Scratches(ImageFilter):
    name = "scratches"

    def __init__(self, n_scratches: int = 8, noise_sigma: float = 12.0,
                 seed: int = 42, dust: float = 0.004,
                 vertical_bias: float = 0.4,
                 texture_strength: float = 0.35):
        self.n_scratches = int(n_scratches)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)
        self.dust = float(max(0.0, dust))
        self.vertical_bias = float(np.clip(vertical_bias, 0.0, 1.0))
        self.texture_strength = float(np.clip(texture_strength, 0.0, 1.0))

    #текстуры
    def _apply_textures(self, out):
        h, w = out.shape[:2]
        
        tex = _load_texture("scratches.png", (h, w))
        if tex is not None:
            print(">>> УСПЕХ: Текстура scratches.png загружена!") # <-- Добавь это
            a = self.texture_strength
            out *= (1.0 - a) + a * tex[..., None]
        else:
            print(">>> ОШИБКА: Файл scratches.png НЕ НАЙДЕН!") # <-- Добавь это

        dust_tex = _load_texture("dust.png", (h, w))
        if dust_tex is not None:
            print(">>> УСПЕХ: Текстура dust.png загружена!") # <-- Добавь это
            a = self.texture_strength * 0.5
            out = out + (1.0 - out / 255.0) * dust_tex[..., None] * a * 255.0
        else:
            print(">>> ОШИБКА: Файл dust.png НЕ НАЙДЕН!") # <-- Добавь это

        grain_tex = _load_texture("grain.png", (h, w))
        if grain_tex is not None:
            print(">>> УСПЕХ: Текстура grain.png загружена!") # <-- Добавь это
            a = self.texture_strength * 0.3
            out += (grain_tex[..., None] - 0.5) * a * 40.0
        else:
            print(">>> ОШИБКА: Файл grain.png НЕ НАЙДЕН!") # <-- Добавь это

        return out

    #хаотичная царапина (ломаная)
    def _draw_scratch(self, out, rng, h, w):
        x = float(rng.integers(0, w))
        y = float(rng.integers(0, h))
        length = int(rng.integers(max(20, h // 6), max(40, h)))

        if rng.random() < self.vertical_bias:
            ang = float(rng.uniform(-0.25, 0.25)) + np.pi / 2
        else:
            ang = float(rng.uniform(0, 2 * np.pi))

        base_alpha = float(rng.uniform(0.08, 0.30))
        color = float(rng.uniform(190.0, 235.0))
        thickness = 1 if rng.random() < 0.85 else 2

        step = 1.0
        for t in range(length):
            ang += float(rng.normal(0.0, 0.06))   # блуждание угла
            x += np.cos(ang) * step
            y += np.sin(ang) * step
            xi, yi = int(x), int(y)
            if not (0 <= xi < w and 0 <= yi < h):
                break

            fade = 1.0 - t / length
            a = base_alpha * fade * float(rng.uniform(0.6, 1.0))

            xs = slice(xi, min(w, xi + thickness))
            out[yi, xs] = color * a + out[yi, xs] * (1.0 - a)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        h, w = image.shape[:2]
        out = image.astype(np.float32)

        # 1) зерно
        if self.noise_sigma > 0:
            out += rng.normal(0, self.noise_sigma, out.shape).astype(np.float32)

        # 2) готовые текстуры (если есть)
        out = self._apply_textures(out)

        # 3) мягкая пыль — пятна, а не бинарные точки
        if self.dust > 0:
            n_spots = int(self.dust * h * w * 0.02)
            for _ in range(n_spots):
                cy = int(rng.integers(0, h))
                cx = int(rng.integers(0, w))
                r = int(rng.integers(1, 3))
                bright = rng.random() < 0.5
                val = (float(rng.uniform(200, 245)) if bright
                       else float(rng.uniform(10, 50)))
                y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                patch = out[y0:y1, x0:x1]
                out[y0:y1, x0:x1] = patch * 0.6 + val * 0.4

        # 4) хаотичные тонкие царапины
        for _ in range(self.n_scratches):
            self._draw_scratch(out, rng, h, w)

        return self._u8(out)