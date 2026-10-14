# # import numpy as np
# # from .base import ImageFilter


# # class ScratchesNoise(ImageFilter):
# # #Состаривание: царапины + шум

# #     def __init__(self, n_scratches: int = 25, noise_std: float = 10.0,
# #                  sp_amount: float = 0.02):
# #         self.n_scratches = n_scratches
# #         self.noise_std = noise_std
# #         self.sp_amount = sp_amount

# #     def apply_filter(self, image: np.ndarray) -> np.ndarray:
# #         img = image.astype(np.float32)
# #         if img.ndim == 2: #1канал в 3х
# #             img = np.stack([img] * 3, axis=2) #три копии в 3ю ось

# #         h, w = img.shape[:2]

# #         for _ in range(self.n_scratches): #царапины
# #             x = np.random.randint(0, w)
# #             y0 = np.random.randint(0, h)
# #             length = np.random.randint(h // 8, h)
# #             y1 = min(h - 1, y0 + length)
# #             thickness = np.random.randint(1, 3) #ширина
# #             color = 255 if np.random.rand() > 0.5 else 0
# #             img[y0:y1, max(0, x - thickness):x + thickness] = color

# #         if self.noise_std > 0: #шум
# #             img += np.random.normal(0, self.noise_std, img.shape)

# #         if self.sp_amount > 0: #импульсивный шум
# #             n = int(self.sp_amount * h * w) #кол-во пикселей
# #             #случайные коорд
# #             ys = np.random.randint(0, h, n)
# #             xs = np.random.randint(0, w, n)
# #             vals = np.where(np.random.rand(n) > 0.5, 255, 0) #случ знач
# #             img[ys, xs] = vals[:, None]

# #         return np.clip(img, 0, 255).astype(np.uint8)


import numpy as np
from .base import ImageFilter


class ScratchesNoise(ImageFilter):
    # Состаривание: царапины + шум + эффект антиквариата
    def __init__(self, noise: float = 0.10, scratch_density: float = 0.03, seed: int = 40):
        self.noise = noise
        self.scratch_density = scratch_density
        self.seed = seed

    @staticmethod
    def _draw_line(field: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   value: float, thickness: int) -> None:
        # Рисует линию на 2D-массиве field через линейную интерполяцию.
        n = max(abs(x2 - x1), abs(y2 - y1), 1)
        xs = np.linspace(x1, x2, n).round().astype(np.int32)
        ys = np.linspace(y1, y2, n).round().astype(np.int32)
        h, w = field.shape
        for dx in range(thickness):
            xr = np.clip(xs + dx, 0, w - 1)
            yr = np.clip(ys, 0, h - 1)
            field[yr, xr] = np.maximum(field[yr, xr], value)

    @staticmethod
    def _blur3x3(image: np.ndarray, iterations: int) -> np.ndarray:
        # Размытие усреднением 3x3
        result = image.astype(np.float32)
        for _ in range(iterations):
            p = np.pad(result, 1, mode="edge")
            result = (
                p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
                p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
                p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
            ) / 9.0
        return result

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        rng = np.random.default_rng(self.seed)

        # зерно
        grain = rng.normal(0.0, 255.0 * self.noise * 0.18, size=(h, w, 1))

        # царапины
        stripe_field = np.zeros((h, w), dtype=np.float32)
        num_scratches = int(w * self.scratch_density * 0.5)

        for _ in range(num_scratches):
            x1 = rng.integers(0, w)
            y1 = rng.integers(0, h)
            length = rng.integers(int(h * 0.05), int(h * 0.40))
            angle = rng.uniform(-5, 5) * np.pi / 180.0
            x2 = int(x1 + length * np.sin(angle))
            y2 = int(y1 + length * np.cos(angle))

            thickness = int(rng.choice([1, 1, 1, 2]))
            intensity = float(rng.uniform(0.20, 0.45))

            self._draw_line(stripe_field, x1, y1, x2, y2, intensity, thickness)

        # Размываем царапины
        stripe_field = self._blur3x3(stripe_field, iterations=3)

        # Пятна/Точки — тоже делаем бледнее
        specks = (rng.random((h, w)) < (self.scratch_density * 0.04)).astype(np.float32)
        specks *= 0.4

        # Объед царапины и точки
        scratch = np.clip(stripe_field + specks, 0.0, 1.0)[..., None]
        scratch_value = rng.uniform(0.60, 0.85, size=(h, w, 1))

        # Наложение эффектов на изобр
        result = (image.astype(np.float32) + grain + scratch * (255.0 * scratch_value - image.astype(np.float32)))

        result = np.clip(result, 0, 255).astype(np.uint8)

        # эффект антиквариата (сепия)
        b, g, r = (result[:, :, 0].astype(np.float32),
                   result[:, :, 1].astype(np.float32),
                   result[:, :, 2].astype(np.float32))
        out_r = 0.393 * r + 0.769 * g + 0.189 * b
        out_g = 0.349 * r + 0.686 * g + 0.168 * b
        out_b = 0.272 * r + 0.534 * g + 0.131 * b
        sepia = np.stack((out_b, out_g, out_r), axis=-1)

        strength = 0.6
        final_result = result.astype(np.float32) * (1.0 - strength) + sepia * strength

        return np.clip(final_result, 0, 255).astype(np.uint8)