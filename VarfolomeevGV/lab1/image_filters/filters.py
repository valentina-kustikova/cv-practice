"""
Image filters module using OpenCV
"""
import cv2
import numpy as np
from math import floor
from typing import Tuple, Optional


# TODO: Check all functions and methods for using restricted cv2 functions and replace them.
# TODO: Move apply_overlay* functionlity into methods or move it to separate module.

class ImageFilter:
    """Base class for image filters"""

    @staticmethod
    def apply_linear_resize(image: np.ndarray,
                            new_height: int,
                            new_width: int) -> np.ndarray:
        """Resizes image using linear stretching with index mapping"""
        if image is None:
            raise ValueError("Input image must not be None")

        original_height, original_width = image.shape[:2]
        new_height = max(1, int(new_height))
        new_width = max(1, int(new_width))

        if new_height == original_height and new_width == original_width:
            return image.copy()

        if original_height == 0 or original_width == 0:
            raise ValueError("Input image dimensions must be greater than zero")

        y_indices = np.linspace(0, original_height - 1, new_height, dtype=np.float64)
        x_indices = np.linspace(0, original_width - 1, new_width, dtype=np.float64)

        y_indices = np.clip(np.round(y_indices).astype(np.intp), 0, original_height - 1)
        x_indices = np.clip(np.round(x_indices).astype(np.intp), 0, original_width - 1)

        resized = image[np.ix_(y_indices, x_indices)]
        return np.ascontiguousarray(resized)

    @staticmethod
    def apply_Nearest_Neighbor_interpolation(image: np.ndarray, scale_factor: int) -> np.ndarray:
        old_w = image.shape[1]
        old_h = image.shape[0]

        new_w = floor(scale_factor * old_w)
        new_h = floor(scale_factor * old_h)

        new_img_array = np.zeros((new_h, new_w, 3), np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                x = i / scale_factor
                y = j / scale_factor
                new_img_array[i, j] = image[floor(x), floor(y)]
        return new_img_array

    @staticmethod
    def apply_Bilinear_interpolation(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        if image is None:
            raise ValueError("Input image must not be None")
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        original_height, original_width = image.shape[:2]

        new_height = max(1, int(round(original_height * scale_factor)))
        new_width = max(1, int(round(original_width * scale_factor)))

        if new_height == original_height and new_width == original_width:
            return image.copy()

        working_image = image.astype(np.float32)

        y_coords = np.linspace(0, original_height - 1, new_height, dtype=np.float32)
        x_coords = np.linspace(0, original_width - 1, new_width, dtype=np.float32)

        y0 = np.floor(y_coords).astype(np.intp)
        x0 = np.floor(x_coords).astype(np.intp)
        y1 = np.clip(y0 + 1, 0, original_height - 1)
        x1 = np.clip(x0 + 1, 0, original_width - 1)

        dy = (y_coords - y0).reshape(new_height, 1, 1).astype(np.float32)
        dx = (x_coords - x0).reshape(1, new_width, 1).astype(np.float32)

        top_left = working_image[y0[:, None], x0[None, :], :]
        top_right = working_image[y0[:, None], x1[None, :], :]
        bottom_left = working_image[y1[:, None], x0[None, :], :]
        bottom_right = working_image[y1[:, None], x1[None, :], :]

        top = top_left * (1.0 - dx) + top_right * dx
        bottom = bottom_left * (1.0 - dx) + bottom_right * dx
        interpolated = top * (1.0 - dy) + bottom * dy

        result = np.clip(interpolated, 0, 255).astype(np.uint8)

        return result

    @staticmethod
    def apply_overlay(image: np.ndarray, overlay: Optional[np.ndarray]) -> np.ndarray:
        """Overlays external image using its alpha channel when available (used in frame_curvy and glare filters)"""
        if image is None:
            raise ValueError("Input image must not be None")
        if overlay is None:
            return image.copy()

        base = image
        overlay_image = overlay

        overlay_channels = overlay_image.shape[2]
        base_channels = base.shape[2]

        target_size = (base.shape[1], base.shape[0])
        if overlay_image.shape[:2] != base.shape[:2]:
            overlay_image = ImageFilter.apply_linear_resize(
                overlay_image,
                base.shape[0],
                base.shape[1]
            )

        overlay_alpha_channel: Optional[np.ndarray] = None
        if overlay_channels == 3:
            overlay_bgr = overlay_image.astype(np.float32)
            alpha = np.ones((base.shape[0], base.shape[1], 1), dtype=np.float32)
        else:
            overlay_bgr = overlay_image[..., :3].astype(np.float32)
            alpha_channel = overlay_image[..., 3].astype(np.float32) / 255.0
            alpha = alpha_channel[..., np.newaxis]
            overlay_alpha_channel = alpha_channel

        base_bgr = base[..., :3].astype(np.float32)

        blended = overlay_bgr * alpha + base_bgr * (1.0 - alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if base_channels == 1:
            return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

        if base_channels == 4:
            base_alpha = base[..., 3].astype(np.float32) / 255.0
            if overlay_alpha_channel is not None:
                out_alpha = np.clip(overlay_alpha_channel + (1.0 - overlay_alpha_channel) * base_alpha, 0.0, 1.0)
            else:
                out_alpha = base_alpha
            out_alpha = (out_alpha * 255.0).astype(np.uint8)
            return np.dstack((blended, out_alpha))

        return blended

    @staticmethod
    def apply_overlay_centered(image: np.ndarray,
                               overlay: Optional[np.ndarray],
                               center_x: int,
                               center_y: int,
                               intensity: float = 1.0,
                               scale: float = 1.0) -> np.ndarray:
        """Overlays external image centered at given coordinates without stretching"""
        if image is None:
            raise ValueError("Input image must not be None")
        if overlay is None:
            return image.copy()

        if intensity <= 0:
            return image.copy()

        base = image.astype(np.float32)
        h, w = base.shape[:2]

        overlay_image = overlay
        if overlay_image.shape[2] == 3:
            alpha_plain = np.full(overlay_image.shape[:2], 255, dtype=overlay_image.dtype)
            overlay_image = np.dstack((overlay_image, alpha_plain))

        overlay_image = overlay_image.astype(np.float32)

        if scale != 1.0 and scale > 0:
            new_width = max(1, int(round(overlay_image.shape[1] * scale)))
            new_height = max(1, int(round(overlay_image.shape[0] * scale)))
            overlay_image = ImageFilter.apply_linear_resize(overlay_image, new_height, new_width)

        overlay_h, overlay_w = overlay_image.shape[:2]

        center_x = int(np.clip(center_x, 0, w - 1))
        center_y = int(np.clip(center_y, 0, h - 1))

        half_w = overlay_w // 2
        half_h = overlay_h // 2

        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = x1 + overlay_w
        y2 = y1 + overlay_h

        base_x1 = max(0, x1)
        base_y1 = max(0, y1)
        base_x2 = min(w, x2)
        base_y2 = min(h, y2)

        overlay_x1 = base_x1 - x1
        overlay_y1 = base_y1 - y1
        overlay_x2 = overlay_x1 + (base_x2 - base_x1)
        overlay_y2 = overlay_y1 + (base_y2 - base_y1)

        if base_x1 >= base_x2 or base_y1 >= base_y2:
            return image.copy()

        overlay_region = overlay_image[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        overlay_rgb = overlay_region[..., :3]
        overlay_alpha = (overlay_region[..., 3] / 255.0) * np.clip(intensity, 0.0, 1.0)

        if base.ndim == 2:
            base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        else:
            base_rgb = base.copy()

        base_subregion = base_rgb[base_y1:base_y2, base_x1:base_x2]

        blended_region = overlay_rgb * overlay_alpha[..., np.newaxis] + base_subregion * (1.0 - overlay_alpha[..., np.newaxis])
        base_rgb[base_y1:base_y2, base_x1:base_x2] = blended_region

        result = np.clip(base_rgb, 0, 255).astype(np.uint8)

        if base.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        return result

    @staticmethod
    def apply_sepia(image: np.ndarray) -> np.ndarray:
        """Applies sepia effect"""
        if image is None:
            raise ValueError("Input image must not be None")

        has_alpha = image.ndim == 3 and image.shape[2] == 4

        image_bgr = image[..., :3]

        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ], dtype=np.float32)

        transformed = image_bgr.astype(np.float32) @ sepia_matrix.T
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)

        return transformed

    @staticmethod
    def apply_vignette(image: np.ndarray,
                       intensity: float = 0.7,
                       radius: float = 0.5,
                       center_x: Optional[int] = None,
                       center_y: Optional[int] = None) -> np.ndarray:
        """Applies vignette effect"""
        if image is None:
            raise ValueError("Input image must not be None")

        intensity = float(np.clip(intensity, 0.0, 1.0))
        radius = float(np.clip(radius, 0.01, 1.0))

        if intensity == 0:
            return image.copy()

        height, width = image.shape[:2]
        cx = float(center_x) if center_x is not None else width / 2.0
        cy = float(center_y) if center_y is not None else height / 2.0

        cx = float(np.clip(cx, 0.0, width - 1))
        cy = float(np.clip(cy, 0.0, height - 1))

        y, x = np.indices((height, width), dtype=np.float32)
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        distances_to_corners = np.array([
            np.sqrt((0 - cx) ** 2 + (0 - cy) ** 2),
            np.sqrt((0 - cx) ** 2 + (height - 1 - cy) ** 2),
            np.sqrt((width - 1 - cx) ** 2 + (0 - cy) ** 2),
            np.sqrt((width - 1 - cx) ** 2 + (height - 1 - cy) ** 2),
        ])
        max_distance = distances_to_corners.max()
        max_radius = radius * max_distance if max_distance > 0 else 1.0

        normalized = distances / max_radius
        normalized = np.clip(normalized, 0.0, 1.0)

        mask = 1.0 - intensity * normalized
        mask = mask[..., np.newaxis] if image.ndim == 3 else mask

        result = image.astype(np.float32) * mask
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_pixelation(image: np.ndarray,
                        x: int, y: int, width: int, height: int,
                        pixel_size: int = 10) -> np.ndarray:
        """Pixelates rectangular area"""
        if image is None:
            raise ValueError("Input image must not be None")

        h, w = image.shape[:2]
        pixel_size = max(1, int(pixel_size))

        # Допуская плохие входные данные, приводим координаты к допустимым диапазонам.
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        width = max(1, int(width))
        height = max(1, int(height))

        # Гарантия того, что правая и нижняя границы не выходят за пределы изображения
        # а также регион содержит минимум один пиксель.
        x_end = max(x + 1, min(w, x + width))
        y_end = max(y + 1, min(h, y + height))

        roi = image[y:y_end, x:x_end]
        if roi.size == 0:
            return image.copy()

        roi_h, roi_w = roi.shape[:2]
        down_w = max(1, roi_w // pixel_size)
        down_h = max(1, roi_h // pixel_size)

        reduced = np.zeros((down_h, down_w, roi.shape[2]), dtype=np.float32)

        block_h = roi_h / down_h
        block_w = roi_w / down_w

        for i in range(down_h):    # усредняем блоки для понижения разрешения
            for j in range(down_w):
                y_start = int(round(i * block_h))
                y_block_end = int(round((i + 1) * block_h))
                x_start = int(round(j * block_w))
                x_block_end = int(round((j + 1) * block_w))

                y_block_end = min(y_block_end, roi_h)
                x_block_end = min(x_block_end, roi_w)
                if y_block_end <= y_start:
                    y_block_end = min(y_start + 1, roi_h)
                if x_block_end <= x_start:
                    x_block_end = min(x_start + 1, roi_w)
                y_start = max(0, min(y_start, roi_h - 1))
                x_start = max(0, min(x_start, roi_w - 1))

                block = roi[y_start:y_block_end, x_start:x_block_end]
                if block.size == 0:
                    continue
                reduced[i, j] = block.mean(axis=(0, 1))

        # zeros_like, чтобы поддерживать BGR и Grayscale.
        pixelated = np.zeros_like(roi)
        for i in range(roi_h):    # растягиваем усреднённые блоки обратно
            for j in range(roi_w):
                src_i = int(i / block_h)
                src_j = int(j / block_w)
                src_i = min(src_i, down_h - 1)
                src_j = min(src_j, down_w - 1)
                pixelated[i, j] = reduced[src_i, src_j]

        result = image.copy()
        result[y:y_end, x:x_end] = np.clip(pixelated, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def apply_frame_simple(image: np.ndarray,
                          frame_width: int = 10,
                          color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Applies simple rectangular frame"""
        if image is None:
            raise ValueError("Input image must not be None")

        frame_width = max(0, int(frame_width))
        if frame_width == 0:
            return image.copy()

        color = tuple(int(c) % 256 for c in color)
        if len(color) == 3:
            color = (color[2], color[1], color[0])
        result = image.copy()
        h, w = image.shape[:2]

        fw = min(frame_width, h // 2 + (h % 2), w // 2 + (w % 2))

        # верхняя полоса
        result[:fw, :] = color
        # нижняя полоса
        result[h - fw:h, :] = color
        # левая полоса
        result[:, :fw] = color
        # правая полоса
        result[:, w - fw:w] = color

        return result

    @staticmethod
    def apply_frame_curvy(image: np.ndarray,
                         frame_width: int = 10,
                         frame_type: str = "wave",
                         color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """Applies decorative curved frame"""
        if image is None:
            raise ValueError("Input image must not be None")

        frame_width = max(0, int(frame_width))
        if frame_width == 0:
            return image.copy()

        color = tuple(int(c) % 256 for c in color)
        if len(color) == 3:
            color = (color[2], color[1], color[0])
        h, w = image.shape[:2]
        overlay = image.copy()

        if frame_type == "wave":
            amplitude = max(2, frame_width)
            period = max(20, min(w, h) // 4)

            top_points = []
            bottom_points = []
            for x in range(w):
                offset = amplitude * 0.5 * (1 + np.sin(2 * np.pi * x / period))
                top_points.append((x, int(offset)))
                bottom_points.append((x, h - 1 - int(offset)))

            top_points = np.array(top_points, dtype=np.int32)
            bottom_points = np.array(bottom_points, dtype=np.int32)

            left_points = []
            right_points = []
            for y in range(h):
                offset = amplitude * 0.5 * (1 + np.sin(2 * np.pi * y / period))
                left_points.append((int(offset), y))
                right_points.append((w - 1 - int(offset), y))

            left_points = np.array(left_points, dtype=np.int32)
            right_points = np.array(right_points, dtype=np.int32)

            cv2.polylines(overlay, [top_points], False, color, thickness=frame_width)
            cv2.polylines(overlay, [bottom_points], False, color, thickness=frame_width)
            cv2.polylines(overlay, [left_points], False, color, thickness=frame_width)
            cv2.polylines(overlay, [right_points], False, color, thickness=frame_width)
        else:
            cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), color, thickness=frame_width)

        return overlay

    @staticmethod
    def apply_glare(image: np.ndarray,
                   center_x: int, center_y: int,
                   radius: int = 100,
                   intensity: float = 0.5) -> np.ndarray:
        """Applies lens glare effect"""
        if image is None:
            raise ValueError("Input image must not be None")

        intensity = float(np.clip(intensity, 0.0, 1.0))
        radius = max(1, int(radius))

        h, w = image.shape[:2]
        center_x = int(np.clip(center_x, 0, w - 1))
        center_y = int(np.clip(center_y, 0, h - 1))

        y, x = np.indices((h, w), dtype=np.float32)
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        mask = np.clip(1.0 - (distances / radius), 0.0, 1.0)
        mask = mask ** 2  # smoother falloff
        if image.ndim == 3:
            mask = mask[..., np.newaxis]

        flare = mask * intensity * 255.0
        result = image.astype(np.float32) + flare
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_watercolor_texture(image: np.ndarray,
                                texture_intensity: float = 0.3,
                                texture_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies watercolor paper texture using external overlay"""
        if image is None:
            raise ValueError("Input image must not be None")

        texture_intensity = float(np.clip(texture_intensity, 0.0, 1.0))
        if texture_intensity == 0 or texture_image is None:
            raise ValueError("Texture overlay is required for watercolor texture")

        smooth = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
        h, w = image.shape[:2]

        overlay = texture_image

        if overlay.ndim == 2:
            texture_bgr = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            alpha_channel = None
        elif overlay.shape[2] == 3:
            texture_bgr = overlay[..., :3]
            alpha_channel = None
        else:
            texture_bgr = overlay[..., :3]
            alpha_channel = overlay[..., 3].astype(np.float32) / 255.0

        if texture_bgr.shape[:2] != (h, w):
            texture_bgr = ImageFilter.apply_linear_resize(texture_bgr, h, w)
            if alpha_channel is not None:
                alpha_channel = ImageFilter.apply_linear_resize(alpha_channel[..., np.newaxis], h, w)[..., 0]

        texture_bgr = texture_bgr.astype(np.float32)

        if alpha_channel is None:
            alpha_channel = cv2.cvtColor(texture_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        alpha_map = np.clip(alpha_channel * texture_intensity, 0.0, 1.0)
        if image.ndim == 3:
            alpha_map = alpha_map[..., np.newaxis]

        if image.ndim == 2:
            smooth_bgr = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR).astype(np.float32)
        else:
            smooth_bgr = smooth.astype(np.float32)

        blended = smooth_bgr * (1.0 - alpha_map) + texture_bgr * alpha_map
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

        return blended
