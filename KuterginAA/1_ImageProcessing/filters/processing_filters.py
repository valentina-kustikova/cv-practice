import cv2
import numpy as np
from .base import ImageFilter


class ResizeFilter(ImageFilter):

    def __init__(self, target_width: int, target_height: int):
        self.target_width = int(target_width)
        self.target_height = int(target_height)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        orig_height, orig_width = image.shape[:2]
        row_indices = (
            np.linspace(0, orig_height - 1, self.target_height)
            .round()
            .astype(int)
        )
        col_indices = (
            np.linspace(0, orig_width - 1, self.target_width)
            .round()
            .astype(int)
        )
        return image[np.ix_(row_indices, col_indices)]


class GrayScaleFilter(ImageFilter):

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.dstack([gray, gray, gray])


class AntiqueFilter(ImageFilter):

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        sepia_matrix = np.array(
            [[0.131, 0.543, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]]
        )
        sepia_img = np.dot(image.astype(np.float32), sepia_matrix.T)

        h, w = image.shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        vignette_mask = 1 - 0.5 * (dist / max_dist) ** 2

        out = sepia_img * vignette_mask[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)


class FadeColorFilter(ImageFilter):

    def __init__(self, factor: float = 0.5):
        self.factor = float(factor)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        faded = image.astype(np.float32) * self.factor + 128 * (1 - self.factor)
        faded[:, :, 0] = faded[:, :, 0] * 0.9 + 25
        return np.clip(faded, 0, 255).astype(np.uint8)


class InfraredFilmFilter(ImageFilter):

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        ir_r = np.clip(r * 1.5 + g * 0.5, 0, 255)
        ir_g = np.clip(g * 0.3 + b * 0.2, 0, 255)
        ir_b = np.clip(b * 0.5, 0, 255)

        return np.dstack([ir_b, ir_g, ir_r]).astype(np.uint8)


class MatteFilter(ImageFilter):

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h / 2.0, w / 2.0

        y, x = np.ogrid[:h, :w]
        ellipse_equation = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2

        out = image.copy()
        out[ellipse_equation > 1.0] = [255, 255, 255]
        return out


class OldPhotoNoiseFilter(ImageFilter):

    def __init__(self, noise_amount: float = 0.02, scratch_count: int = 5):
        self.noise_amount = float(noise_amount)
        self.scratch_count = int(scratch_count)

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        h, w = image.shape[:2]

        num_noise = int(self.noise_amount * h * w)
        ys = np.random.randint(0, h, num_noise // 2)
        xs = np.random.randint(0, w, num_noise // 2)
        out[ys, xs] = [255, 255, 255]

        ys = np.random.randint(0, h, num_noise // 2)
        xs = np.random.randint(0, w, num_noise // 2)
        out[ys, xs] = [0, 0, 0]

        for _ in range(self.scratch_count):
            x_pos = np.random.randint(0, w)
            y_start = np.random.randint(0, h // 2)
            y_end = np.random.randint(y_start, h)
            thickness = np.random.randint(1, 2)
            out[
                y_start:y_end,
                max(0, x_pos - thickness) : min(w, x_pos + thickness),
            ] = [220, 220, 220]

        return out


class NeonFilter(ImageFilter):

    def apply_filter(self, image: np.ndarray) -> np.ndarray:
        gray = GrayScaleFilter().apply_filter(image)[:, :, 0].astype(np.float32)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray, -1, sobel_x)
        grad_y = cv2.filter2D(gray, -1, sobel_y)
        magnitude = np.hypot(grad_x, grad_y)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
        neon_b = np.clip(magnitude * 1.5, 0, 255).astype(np.uint8)
        neon_g = np.clip(magnitude * 0.3, 0, 255).astype(np.uint8)
        neon_r = np.clip(magnitude * 1.2, 0, 255).astype(np.uint8)

        return np.dstack([neon_b, neon_g, neon_r])