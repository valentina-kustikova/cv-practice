import numpy as np


class Filters:
    @staticmethod
    def _bilinear_interpolation(image, x, y):
        x1 = np.floor(x).astype(int)
        y1 = np.floor(y).astype(int)
        x2 = np.minimum(x1 + 1, image.shape[1] - 1)
        y2 = np.minimum(y1 + 1, image.shape[0] - 1)
        dx = x - x1
        dy = y - y1
        if len(image.shape) == 3:
            dx = dx[..., np.newaxis]
            dy = dy[..., np.newaxis]
        f11 = image[y1, x1]
        f12 = image[y1, x2]
        f21 = image[y2, x1]
        f22 = image[y2, x2]
        result = (f11 * (1 - dx) * (1 - dy) +
                  f12 * dx * (1 - dy) +
                  f21 * (1 - dx) * dy +
                  f22 * dx * dy)

        return result

    @staticmethod
    def _nearest_neighbor_interpolation(image, x, y):
        x = np.floor(x).astype(int)
        y = np.floor(y).astype(int)
        src_x = np.minimum(x, image.shape[1] - 1)
        src_y = np.minimum(y, image.shape[0] - 1)
        dst = image[src_y, src_x]

        return dst

    @staticmethod
    def resize_image(image, new_size, interpolation=_bilinear_interpolation):
        h, w = image.shape[:2]
        new_w, new_h = new_size
        j = np.arange(new_w)
        i = np.arange(new_h)
        x = j * (w - 1) / max(new_w - 1, 1)
        y = i * (h - 1) / max(new_h - 1, 1)
        X, Y = np.meshgrid(x, y)
        resized = interpolation(image, X, Y)
        resized = np.clip(resized, 0, 255).astype(np.uint8)

        return resized

    @staticmethod
    def sepia(image):
        intensity = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
        k = 25
        sepia_image = np.zeros_like(image)
        sepia_image[:, :, 2] = np.clip(intensity + 2 * k, 0, 255)
        sepia_image[:, :, 1] = np.clip(intensity + 0.5 * k, 0, 255)
        sepia_image[:, :, 0] = np.clip(intensity - k, 0, 255)

        return sepia_image

    @staticmethod
    def _gaussian_kernel(ksize, sigma):
        if sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        i = np.arange(ksize)
        center = (ksize - 1) / 2
        x = i - center
        kernel = np.exp(-(x * x) / (2 * sigma * sigma))
        kernel /= kernel.sum()

        return kernel

    @staticmethod
    def vignette(image):
        h, w = image.shape[:2]
        sigma = max(h, w) * 0.3
        x_kernel = Filters._gaussian_kernel(w, sigma)
        y_kernel = Filters._gaussian_kernel(h, sigma)
        res_kernel = np.outer(y_kernel, x_kernel)
        mask = res_kernel / res_kernel.max()
        vignette_image = np.copy(image)
        if len(image.shape) == 3:
            vignette_image = (vignette_image * mask[:, :, np.newaxis]).astype(np.uint8)
        else:
            vignette_image = vignette_image * mask
        vignette_image = np.clip(vignette_image, 0, 255)

        return vignette_image

    @staticmethod
    def pixelation(image):
        height, width = image.shape[:2]
        pixel_size = 8
        new_width = width // pixel_size
        new_height = height // pixel_size
        small = Filters.resize_image(image, (new_width, new_height))
        pixel_image = Filters.resize_image(small, (width, height), interpolation=Filters._nearest_neighbor_interpolation)

        return pixel_image

    @staticmethod
    def pixelation_of_roi(image, x1, y1, x2, y2):
        image_roi = image[y1:y2, x1:x2]
        pixel_image_roi = Filters.pixelation(image_roi)
        res_pixel_image = np.copy(image)
        res_pixel_image[y1:y2, x1:x2] = pixel_image_roi

        return res_pixel_image

    @staticmethod
    def rectangular_frame(image, frame_width=30, frame_color=(0, 0, 255)):
        if frame_width < 0:
            frame_width = -frame_width
        color_image = np.zeros_like(image)
        color_image[:, :, 0], color_image[:, :, 1], color_image[:, :, 2] = frame_color
        color_image[frame_width:image.shape[0] - frame_width, frame_width:image.shape[1] - frame_width] = image[
            frame_width:image.shape[0] - frame_width, frame_width:image.shape[1] - frame_width]

        return color_image

    @staticmethod
    def _wavy_frame_mask(mask, frame_width, amplitude, frequency):
        height, width = mask.shape
        y_coords, x_coords = np.indices((height, width))
        top_wave = (amplitude * np.sin(frequency * x_coords)).astype(int)
        top_boundary = np.clip(frame_width + top_wave, 0, height)
        top_mask = y_coords < top_boundary
        mask[top_mask] = True

        bottom_wave = (amplitude * np.sin(frequency * x_coords)).astype(int)
        bottom_boundary = np.clip(height - frame_width + bottom_wave, 0, height)
        bottom_mask = y_coords >= bottom_boundary
        mask[bottom_mask] = True

        left_wave = (amplitude * np.sin(frequency * y_coords)).astype(int)
        left_boundary = np.clip(frame_width + left_wave, 0, width)
        left_mask = x_coords < left_boundary
        mask[left_mask] = True

        right_wave = (amplitude * np.sin(frequency * y_coords)).astype(int)
        right_boundary = np.clip(width - frame_width + right_wave, 0, width)
        right_mask = x_coords >= right_boundary
        mask[right_mask] = True

    @staticmethod
    def _zigzag_frame_mask(mask, frame_width, pattern_size):
        height, width = mask.shape
        y_coords, x_coords = np.indices((height, width))
        segment_pos = (x_coords % pattern_size) / pattern_size
        zigzag_height = np.where(
            segment_pos < 0.5,
            frame_width * (segment_pos / 0.5),
            frame_width * ((1 - segment_pos) / 0.5))
        top_mask = (y_coords < zigzag_height) & (y_coords < frame_width * 1.5)
        mask[top_mask] = True
        bottom_zigzag_height = height - frame_width * np.where(
            segment_pos < 0.5,
            (segment_pos / 0.5),
            ((1 - segment_pos) / 0.5))
        bottom_mask = (y_coords >= bottom_zigzag_height) & (y_coords > height - frame_width * 1.5)
        mask[bottom_mask] = True
        mask[:, 0: frame_width] = True
        mask[:, width - frame_width: width + 1] = True

    @staticmethod
    def _diagonal_frame_mask(mask, frame_width, pattern_size):
        height, width = mask.shape
        y_coords, x_coords = np.indices((height, width))
        diagonal_value = (x_coords + y_coords) % pattern_size
        diagonal_mask = diagonal_value < frame_width
        edge_mask = (
                (x_coords < frame_width * 2) |
                (x_coords >= width - frame_width * 2) |
                (y_coords < frame_width * 2) |
                (y_coords >= height - frame_width * 2))
        mask[diagonal_mask & edge_mask] = True

    @staticmethod
    def figured_frame(image, frame_type="", frame_width=30, frame_color=(0, 0, 255),
                      amplitude=10, frequency=0.1, pattern_size=20):
        height, width = image.shape[:2]
        image_with_frame = image.copy()
        frame_mask = np.zeros((height, width), dtype=bool)
        if frame_type == "wavy":
            Filters._wavy_frame_mask(frame_mask, frame_width, amplitude, frequency)
        elif frame_type == "zigzag":
            Filters._zigzag_frame_mask(frame_mask, frame_width, pattern_size)
        else:
            Filters._diagonal_frame_mask(frame_mask, frame_width, pattern_size)
        image_with_frame[frame_mask] = frame_color

        return image_with_frame

    @staticmethod
    def _create_flare(size, color, intensity):
        flare = np.zeros((size, size, 3), dtype=np.float32)
        center = size // 2
        y_coords, x_coords = np.indices((size, size))
        distances = np.sqrt((x_coords - center) ** 2 + (y_coords - center) ** 2)
        circle_mask = distances < center
        sigma = center / 2
        weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        color_array = np.array(color, dtype=np.float32) * intensity / 255.0
        flare[circle_mask] = color_array * weights[circle_mask, np.newaxis]

        return flare

    @staticmethod
    def _mixing(src1, alpha, src2, beta):
        src1_float = src1.astype(np.float32)
        src2_float = src2.astype(np.float32)
        result = src1_float * alpha + src2_float * beta
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result

    @staticmethod
    def lens_flare(image):
        flare_image = image.copy()
        h, w = image.shape[:2]
        center_x, center_y = w // 2 - w // 5, h // 2 + h // 5
        min_dim = min(w, h)
        flare_size = min_dim // 4
        flare_color = (255, 255, 255)
        intensity = 0.6
        flare_mask = np.zeros((h, w, 3), dtype=np.float32)
        flare = Filters._create_flare(flare_size, flare_color, intensity)
        flare_h, flare_w = flare.shape[:2]
        x1 = center_x - flare_w // 2
        y1 = center_y - flare_h // 2
        x2 = x1 + flare_w
        y2 = y1 + flare_h
        flare_cropped = flare.copy()
        if x1 < 0:
            flare_cropped = flare_cropped[:, -x1:]
            x1 = 0
        if y1 < 0:
            flare_cropped = flare_cropped[-y1:, :]
            y1 = 0
        if x2 > w:
            flare_cropped = flare_cropped[:, :w - x1]
        if y2 > h:
            flare_cropped = flare_cropped[:h - y1, :]

        if flare_cropped.size > 0 and x1 < w and y1 < h:
            actual_h, actual_w = flare_cropped.shape[:2]
            flare_mask[y1:y1 + actual_h, x1:x1 + actual_w] = flare_cropped
        flare_mask = np.clip(flare_mask, 0, 1)
        flare_image = Filters._mixing(flare_image, 1.0, (flare_mask * 255).astype(np.uint8), 1.0)

        return flare_image

    @staticmethod
    def paper_texture(image):
        h, w = image.shape[:2]
        paper = np.full((h, w, 3), [220, 235, 240], dtype=np.uint8)
        noise = np.random.randint(-80, 80, (h, w, 3), dtype=np.int16)
        paper = np.clip(paper.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = Filters._mixing(image, 0.7, paper, 0.3)

        return result
