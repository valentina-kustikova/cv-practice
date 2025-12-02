"""
Image filters module using numpy
"""
import numpy as np
from math import floor
from typing import Tuple, Optional

from .processors import ColorConverter


def _draw_line_numpy(image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                     color: Tuple[int, int, int], thickness: int):
    """Draws a line on image using numpy"""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = image.shape[:2]
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    half_thickness = thickness // 2
    
    while True:
        # Draw thick line by filling a square around each point
        for dy_offset in range(-half_thickness, half_thickness + 1):
            for dx_offset in range(-half_thickness, half_thickness + 1):
                px = x + dx_offset
                py = y + dy_offset
                if 0 <= px < w and 0 <= py < h:
                    if image.ndim == 2:
                        # For grayscale, use average of color channels or first channel
                        gray_value = int(np.mean(color)) if len(color) > 0 else 0
                        image[py, px] = gray_value
                    else:
                        image[py, px] = color
        
        if x == x2 and y == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _draw_polyline_numpy(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int):
    """Draws a polyline on image using numpy"""
    if len(points) < 2:
        return
    
    for i in range(len(points) - 1):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])
        _draw_line_numpy(image, pt1, pt2, color, thickness)


def _draw_rectangle_numpy(image: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                         color: Tuple[int, int, int], thickness: int):
    """Draws a rectangle on image using numpy"""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = image.shape[:2]
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    
    if thickness > 0:
        # Draw four edges
        # Top edge
        for x in range(x1, x2 + 1):
            for t in range(thickness):
                y = y1 + t
                if 0 <= y < h and 0 <= x < w:
                    if image.ndim == 2:
                        image[y, x] = color[0] if len(color) > 0 else 0
                    else:
                        image[y, x] = color
        
        # Bottom edge
        for x in range(x1, x2 + 1):
            for t in range(thickness):
                y = y2 - t
                if 0 <= y < h and 0 <= x < w:
                    if image.ndim == 2:
                        image[y, x] = color[0] if len(color) > 0 else 0
                    else:
                        image[y, x] = color
        
        # Left edge
        for y in range(y1, y2 + 1):
            for t in range(thickness):
                x = x1 + t
                if 0 <= y < h and 0 <= x < w:
                    if image.ndim == 2:
                        image[y, x] = color[0] if len(color) > 0 else 0
                    else:
                        image[y, x] = color
        
        # Right edge
        for y in range(y1, y2 + 1):
            for t in range(thickness):
                x = x2 - t
                if 0 <= y < h and 0 <= x < w:
                    if image.ndim == 2:
                        image[y, x] = color[0] if len(color) > 0 else 0
                    else:
                        image[y, x] = color


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

        # Валидация и приведение координат к допустимым диапазонам
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        width = max(1, int(width))
        height = max(1, int(height))

        # Вычисляем правую и нижнюю границы, не выходящие за пределы изображения
        x_end = min(x + width, w)
        y_end = min(y + height, h)

        # Копируем изображение для модификации
        result = image.copy()

        # Обрабатываем регион блоками pixel_size x pixel_size
        for i in range(y, y_end, pixel_size):
            for j in range(x, x_end, pixel_size):
                # Определяем границы текущего блока с учётом края изображения
                i_end = min(i + pixel_size, y_end)
                j_end = min(j + pixel_size, x_end)

                block = result[i:i_end, j:j_end]
                if block.size == 0:
                    continue

                # Вычисляем среднее значение блока
                # Для цветных изображений: axis=(0,1), для grayscale: просто mean()
                if block.ndim == 3:
                    avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                else:
                    avg_color = block.mean().astype(np.uint8)

                # Заменяем весь блок на среднее значение
                result[i:i_end, j:j_end] = avg_color

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

        # top stripe
        result[:fw, :] = color
        # bottom stripe
        result[h - fw:h, :] = color
        # left stripe
        result[:, :fw] = color
        # right stripe
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

            _draw_polyline_numpy(overlay, top_points, color, frame_width)
            _draw_polyline_numpy(overlay, bottom_points, color, frame_width)
            _draw_polyline_numpy(overlay, left_points, color, frame_width)
            _draw_polyline_numpy(overlay, right_points, color, frame_width)
        else:
            _draw_rectangle_numpy(overlay, (0, 0), (w - 1, h - 1), color, frame_width)

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

        h, w = image.shape[:2]

        overlay = texture_image

        if overlay.ndim == 2:
            texture_bgr = ColorConverter.gray_to_bgr(overlay)
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
            alpha_channel = ColorConverter.bgr_to_gray(texture_bgr.astype(np.uint8)).astype(np.float32) / 255.0

        alpha_map = np.clip(alpha_channel * texture_intensity, 0.0, 1.0)
        if image.ndim == 3:
            alpha_map = alpha_map[..., np.newaxis]

        if image.ndim == 2:
            image_bgr = ColorConverter.gray_to_bgr(image).astype(np.float32)
        else:
            image_bgr = image.astype(np.float32)

        blended = image_bgr * (1.0 - alpha_map) + texture_bgr * alpha_map
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            return ColorConverter.bgr_to_gray(blended)

        return blended
