import cv2
import numpy as np

class ImageFilter:
    @staticmethod
    def get_filter(filter_type, **kwargs):
        filters = {
            'resize': ResizeFilter,
            'gray': GrayScaleFilter,
            'antique': AntiqueFilter,
            'fade': FadeColorFilter,
            'film': FilmSimulationFilter,
            'matte': MatteFilter,
            'scratch': ScratchNoiseFilter,
            'neon': NeonFilter
        }
        if filter_type not in filters:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return filters[filter_type](**kwargs)

    def apply_filter(self, image):
        raise NotImplementedError("Each filter must implement apply_filter method.")

class ResizeFilter(ImageFilter):
    def __init__(self, scale=None, width=None, height=None):
        self.scale = float(scale) if scale else None
        self.width = int(width) if width else None
        self.height = int(height) if height else None

    def apply_filter(self, image):
        h, w, c = image.shape
        if self.scale:
            new_h, new_w = int(h * self.scale), int(w * self.scale)
        elif self.width and self.height:
            new_h, new_w = self.height, self.width
        else:
            return image

        x_ratio = float(w) / new_w
        y_ratio = float(h) / new_h

        y_indices = np.arange(new_h)
        x_indices = np.arange(new_w)
        
        y_l = (y_ratio * y_indices).astype(int)
        x_l = (x_ratio * x_indices).astype(int)
        
        y_h = np.minimum(y_l + 1, h - 1)
        x_h = np.minimum(x_l + 1, w - 1)

        y_weight = ((y_ratio * y_indices) - y_l)[:, None, None]
        x_weight = ((x_ratio * x_indices) - x_l)[None, :, None]

        a = image[y_l[:, None], x_l, :]
        b = image[y_l[:, None], x_h, :]
        c_val = image[y_h[:, None], x_l, :]
        d = image[y_h[:, None], x_h, :]
        
        pixel_val = (a * (1 - x_weight) * (1 - y_weight) +
                     b * x_weight * (1 - y_weight) +
                     c_val * (1 - x_weight) * y_weight +
                     d * x_weight * y_weight)
                     
        return np.clip(pixel_val, 0, 255).astype(np.uint8)


class GrayScaleFilter(ImageFilter):
    def apply_filter(self, image):
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

class AntiqueFilter(ImageFilter):
    def apply_filter(self, image):
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        
        output_r = r * 0.393 + g * 0.769 + b * 0.189
        output_g = r * 0.349 + g * 0.686 + b * 0.168
        output_b = r * 0.272 + g * 0.534 + b * 0.131
        
        antique = np.stack([output_b, output_g, output_r], axis=-1)
        return np.clip(antique, 0, 255).astype(np.uint8)

class FadeColorFilter(ImageFilter):
    def __init__(self, factor=0.4):
        self.factor = float(factor)

    def apply_filter(self, image):
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        gray_3d = np.stack([gray, gray, gray], axis=-1)
        faded = image * (1 - self.factor) + gray_3d * self.factor
        return np.clip(faded, 0, 255).astype(np.uint8)

class FilmSimulationFilter(ImageFilter):
    def apply_filter(self, image):
        r = image[:, :, 2].astype(float)
        g = image[:, :, 1].astype(float)
        
        ir_gray = r * 0.7 + g * 0.3
        
        ir_gray = (ir_gray - 128) * 1.5 + 128
        ir_gray = np.clip(ir_gray, 0, 255).astype(np.uint8)
        return np.stack([ir_gray, ir_gray, ir_gray], axis=-1)

class MatteFilter(ImageFilter):
    def apply_filter(self, image):
        h, w, c = image.shape
        yc, xc = h / 2.0, w / 2.0
        
        y = np.arange(h)[:, None]
        x = np.arange(w)[None, :]
        
        r1x, r1y = w * 0.45, h * 0.45
        r2x, r2y = w * 0.5, h * 0.5
        
        d_inner = ((x - xc) / r1x) ** 2 + ((y - yc) / r1y) ** 2
        d_outer = ((x - xc) / r2x) ** 2 + ((y - yc) / r2y) ** 2
        
        mask = np.ones((h, w), dtype=float)
        
        inner_mask = d_inner <= 1.0
        outer_mask = d_outer >= 1.0
        transition_mask = (~inner_mask) & (~outer_mask)
        
        norm_d = (d_inner - 1.0) / (d_inner - d_outer + 1e-6)
        
        mask[transition_mask] = 1.0 - np.clip(norm_d[transition_mask], 0, 1)
        mask[outer_mask] = 0.0
        
        mask_3d = mask[:, :, None]
        matte_img = image * mask_3d + 255 * (1.0 - mask_3d)
        return np.clip(matte_img, 0, 255).astype(np.uint8)


class ScratchNoiseFilter(ImageFilter):
    def __init__(self, noise_level=0.05, scratches=5):
        self.noise_level = float(noise_level)
        self.scratches = int(scratches)

    def apply_filter(self, image):
        h, w, c = image.shape
        result = image.copy().astype(float)
        
        noise = np.random.randn(h, w, c) * (self.noise_level * 255)
        result += noise
        
        for _ in range(self.scratches):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            length = np.random.randint(20, 150)
            angle = np.random.uniform(0, 2 * np.pi)
            
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps == 0:
                continue
            x_step = (x2 - x1) / steps
            y_step = (y2 - y1) / steps
            
            for s in range(int(steps)):
                curr_x = int(x1 + s * x_step)
                curr_y = int(y1 + s * y_step)
                if 0 <= curr_x < w and 0 <= curr_y < h:
                    result[curr_y, curr_x, :] = 220
                    
        return np.clip(result, 0, 255).astype(np.uint8)

class NeonFilter(ImageFilter):
    def apply_filter(self, image):
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
        
        h, w = gray.shape
        grad_x = np.zeros_like(gray, dtype=float)
        grad_y = np.zeros_like(gray, dtype=float)
        
        padded = np.pad(gray, 1, mode='edge')
        
        for y in range(1, h + 1):
            for x in range(1, w + 1):
                neighborhood = padded[y-1:y+2, x-1:x+2]
                grad_x[y-1, x-1] = np.sum(neighborhood * sobel_x)
                grad_y[y-1, x-1] = np.sum(neighborhood * sobel_y)
                
        edge = np.sqrt(grad_x**2 + grad_y**2)
        edge = np.clip((edge / edge.max()) * 255 if edge.max() > 0 else 0, 0, 255)
        
        neon = np.zeros((h, w, 3), dtype=np.uint8)
        neon[:, :, 0] = np.clip(edge * 1.0, 0, 255).astype(np.uint8) # Blue
        neon[:, :, 1] = np.clip(edge * 0.8, 0, 255).astype(np.uint8) # Green
        neon[:, :, 2] = np.clip(edge * 0.1, 0, 255).astype(np.uint8) # Red
        return neon
