"""
Image processing classes for filters and overlays
"""
import numpy as np
from typing import Optional, Dict


class ColorConverter:
    """Class for color space conversion operations using numpy"""

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Converts BGR to RGB by reordering channels"""
        return image[..., [2, 1, 0]]

    @staticmethod
    def bgra_to_rgba(image: np.ndarray) -> np.ndarray:
        """Converts BGRA to RGBA by reordering channels"""
        return image[..., [2, 1, 0, 3]]

    @staticmethod
    def bgr_to_gray(image: np.ndarray) -> np.ndarray:
        """Converts BGR to grayscale using standard weights"""
        if image.ndim == 2:
            return image.copy()
        # BGR order: weights for B, G, R channels
        # Standard formula: 0.299*R + 0.587*G + 0.114*B
        # In BGR: image[..., 0] = B, image[..., 1] = G, image[..., 2] = R
        gray = (0.114 * image[..., 0].astype(np.float32) +
                0.587 * image[..., 1].astype(np.float32) +
                0.299 * image[..., 2].astype(np.float32))
        return np.clip(gray, 0, 255).astype(np.uint8)

    @staticmethod
    def gray_to_bgr(image: np.ndarray) -> np.ndarray:
        """Converts grayscale to BGR by duplicating channel"""
        if image.ndim == 3:
            return image.copy()
        return np.repeat(image[..., np.newaxis], 3, axis=-1)


class OverlayProcessor:
    """Class for handling image overlay operations"""

    @staticmethod
    def apply_overlay(image: np.ndarray, overlay: Optional[np.ndarray]) -> np.ndarray:
        """Overlays external image using its alpha channel when available"""
        if image is None:
            raise ValueError("Input image must not be None")
        if overlay is None:
            return image.copy()

        base = image
        overlay_image = overlay

        overlay_channels = overlay_image.shape[2]
        base_channels = base.shape[2]

        if overlay_image.shape[:2] != base.shape[:2]:
            from .filters import ImageFilter
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
            return ColorConverter.bgr_to_gray(blended)

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
            from .filters import ImageFilter
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
            base_rgb = ColorConverter.gray_to_bgr(base)
        else:
            base_rgb = base.copy()

        base_subregion = base_rgb[base_y1:base_y2, base_x1:base_x2]

        blended_region = overlay_rgb * overlay_alpha[..., np.newaxis] + base_subregion * (1.0 - overlay_alpha[..., np.newaxis])
        base_rgb[base_y1:base_y2, base_x1:base_x2] = blended_region

        result = np.clip(base_rgb, 0, 255).astype(np.uint8)

        if base.ndim == 2:
            result = ColorConverter.bgr_to_gray(result)

        return result


class FilterProcessor:
    """Class for processing images with filters and parameters"""

    @staticmethod
    def apply_filter(image: np.ndarray,
                    filter_name: str,
                    filter_params: Dict,
                    overlay_image: Optional[np.ndarray] = None,
                    use_overlay_for_frame: bool = False) -> np.ndarray:
        """Applies filter to image based on filter name and parameters"""
        from .filters import ImageFilter
        
        if filter_name == "Nearest Neighbour Interpolation":
            scale = filter_params.get("scale_factor", 2)
            return ImageFilter.apply_Nearest_Neighbor_interpolation(image, scale)

        elif filter_name == "Bilinear interpolation":
            scale = filter_params.get("scale_factor", 2)
            return ImageFilter.apply_Bilinear_interpolation(image, scale)

        elif filter_name == "Linear resize":
            new_width = filter_params.get("new_width")
            new_height = filter_params.get("new_height")
            if new_width is None or new_height is None:
                h, w = image.shape[:2]
                new_width = w
                new_height = h
            return ImageFilter.apply_linear_resize(image, new_height, new_width)

        elif filter_name == "sepia":
            return ImageFilter.apply_sepia(image)

        elif filter_name == "vignette":
            intensity = filter_params.get("intensity", 70) / 100.0
            radius = filter_params.get("radius", 50) / 100.0
            center_x = filter_params.get("center_x")
            center_y = filter_params.get("center_y")
            if center_x is None or center_y is None:
                height, width = image.shape[:2]
                center_x = width // 2
                center_y = height // 2
            return ImageFilter.apply_vignette(image, intensity, radius, center_x, center_y)

        elif filter_name == "pixelation":
            return ImageFilter.apply_pixelation(
                image,
                filter_params.get("x", 0),
                filter_params.get("y", 0),
                filter_params.get("width", 100),
                filter_params.get("height", 100),
                filter_params.get("pixel_size", 10)
            )

        elif filter_name == "frame_simple":
            color = (
                int(filter_params.get("frame_color_r", 0)) % 256,
                int(filter_params.get("frame_color_g", 0)) % 256,
                int(filter_params.get("frame_color_b", 0)) % 256,
            )
            return ImageFilter.apply_frame_simple(
                image,
                filter_params.get("frame_width", 10),
                color
            )

        elif filter_name == "frame_curvy":
            color = (
                int(filter_params.get("frame_color_r", 0)) % 256,
                int(filter_params.get("frame_color_g", 0)) % 256,
                int(filter_params.get("frame_color_b", 0)) % 256,
            )
            if use_overlay_for_frame and overlay_image is not None:
                return OverlayProcessor.apply_overlay(image, overlay_image)
            return ImageFilter.apply_frame_curvy(
                image,
                filter_params.get("frame_width", 10),
                "wave",
                color
            )

        elif filter_name == "glare":
            h, w = image.shape[:2]
            center_x = filter_params.get("center_x", w // 2)
            center_y = filter_params.get("center_y", h // 2)
            radius = filter_params.get("radius", 100)
            intensity = filter_params.get("intensity", 50) / 100.0
            if overlay_image is not None:
                return OverlayProcessor.apply_overlay_centered(
                    image,
                    overlay_image,
                    center_x,
                    center_y,
                    intensity=intensity,
                    scale=filter_params.get("overlay_scale", 100) / 100.0
                )
            return ImageFilter.apply_glare(image, center_x, center_y, radius, intensity)

        elif filter_name == "watercolor_texture":
            intensity = filter_params.get("texture_intensity", 30) / 100.0
            return ImageFilter.apply_watercolor_texture(image, intensity, overlay_image)

        elif filter_name == "overlay_alpha":
            return OverlayProcessor.apply_overlay(image, overlay_image)

        return image

