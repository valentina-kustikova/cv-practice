from abc import ABC, abstractmethod

import numpy as np
import cv2

class ImageFilter(ABC):
    @abstractmethod
    def apply_filter(self, image):
        pass

class ResizeFilter(ImageFilter):
    def __init__(self, width = 600, height = 800):
        self.width = width
        self.height = height
        
    def apply_filter(self, image):
        h, w = image.shape[:2]
        x_ratio = w / self.width
        y_ratio = h / self.height
        
        x_idx = (np.arange(self.width) * x_ratio).astype(int)
        y_idx = (np.arange(self.height) * y_ratio).astype(int)
        
        return image[np.ix_(y_idx, x_idx)]
        
class GrayscaleFilter(ImageFilter):
    def apply_filter(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        return (0.114 * b + 0.587 * g + 0.229 * r).astype(np.uint8)
    
class AntiqueFilter(ImageFilter):
    def apply_filter(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        new_b = 0.131 * b + 0.534 * g + 0.272 * r
        new_g = 0.168 * b + 0.686 * g + 0.349 * r
        new_r = 0.189 * b + 0.769 * g + 0.393 * r
        
        result = np.stack([new_b, new_g, new_r], axis = 2) ## добавить по углам виньетку (темноту)
        return np.clip(result, 0, 255).astype(np.uint8)
    
class FadeFilter(ImageFilter):
    def __init__ (self, fade_factor = 0.7, brightness = 20):
        self.fade_factor = fade_factor
        self.brightness = brightness
        
    def apply_filter(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        gray = b * 0.114 + g * 0.587 + r * 0.299
        
        new_b = b * self.fade_factor + gray * (1.0 - self.fade_factor) + self.brightness
        new_g = g * self.fade_factor + gray * (1.0 - self.fade_factor) + self.brightness
        new_r = r * self.fade_factor + gray * (1.0 - self.fade_factor) + self.brightness
        
        result = np.stack([new_b, new_g, new_r], axis = 2)
        return np.clip(result, 0, 255).astype(np.uint8)
    
class FilmFilter(ImageFilter):
    def __init__ (self, red_factor = 1, blue_factor = 1):
        self.red_factor = red_factor
        self.blue_factor = blue_factor
        
    def apply_filter(self, image):
        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        new_b = r * self.blue_factor
        new_g = g
        new_r = b * self.red_factor
        
        red_dominant = (new_r > new_g * 1.3) & (new_r > new_b * 1.3)
        new_r = np.where(red_dominant, new_r * 0.6, new_r)
        
        result = np.stack([new_b, new_g, new_r], axis=2)
        return np.clip(result, 0, 255).astype(np.uint8)
    
class MatteFilter(ImageFilter):
    def __init__(self, radius = 0.1, softness=0.5):
        self.radius = radius
        self.softness = softness
        
    def apply_filter(self, image):
        result = image.astype(np.float32)
        h, w = image.shape[:2]
        
        y, x = np.ogrid[:h, :w]
        c_y, c_x = h / 2.0, w / 2.0
        
        dist = np.sqrt(
        (((x - c_x) * (x - c_x)) / ((c_x * self.radius) * (c_x * self.radius))) +
        (((y - c_y) * (y - c_y)) / ((c_y * self.radius) * (c_y * self.radius)))
        )
     
        weight = (dist - 1.0 + self.softness) / self.softness
        weight = np.clip(weight, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        
        weight = weight[..., np.newaxis]
        result = result * (1.0 - weight) + 255.0 * weight
        return np.clip(result, 0, 255).astype(np.uint8)
    

class DamageFilter(ImageFilter):
    def __init__(self, grain = 8, scratch_amount = 55, dust_amount = 110):
        self.grain = grain
        self.scratch_amount = scratch_amount
        self.dust_amount = dust_amount
        
    def apply_filter(self, image):
        result = image.astype(np.float32)
        
        h, w = result.shape[:2]
        
        noise = np.random.normal(0, self.grain, (h, w, 1))     
        color_noise = np.random.normal(0, self.grain * 0.25, image.shape)
        
        result += noise + color_noise
        result = np.clip(result, 0, 255)
        
        
        for i in range(self.scratch_amount):
            x_start = np.random.randint(0, w)
            y_start = np.random.randint(0, h)
            
            length = np.random.randint(10, max(11, int(h * 0.25)))
            angle = np.random.uniform(-0.45, 0.45)
            thickness = np.random.choice([0, 0, 0, 0, 1])
            
            t = np.linspace(0, 1, length)
            
            deviation = np.cumsum(
                np.random.normal(0, 0.2, length)
            )
            
            x = x_start + np.sin(angle) * t * length + deviation
            y = y_start + np.cos(angle) * t * length
            
            keep = np.random.random(length) > 0.15
            
            brightness = np.random.choice([
                np.random.randint(20, 90),
                np.random.randint(180, 256)
            ])
            
            alpha = np.random.uniform(0.15, 0.30)
            
            for offset in range(-thickness, thickness + 1):
                x_pos = np.round(x).astype(int)
                y_pos = np.round(y).astype(int)

                valid = (
                    keep &
                    (x_pos >= 0) & (x_pos < w) &
                    (y_pos >= 0) & (y_pos < h)
                )

                result[y_pos[valid], x_pos[valid]] = (
                    result[y_pos[valid], x_pos[valid]] * (1 - alpha)
                    + brightness * alpha
                )   
            
        y_grid, x_grid = np.ogrid[:h, :w]
            
        for i in range(self.dust_amount):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            radius_x = np.random.choice([1, 1, 1, 2, 2, 3])
            radius_y = np.random.choice([1, 1, 2, 2, 3])
            
            distance = (
                ((x_grid - x) / radius_x) ** 2 +
                ((y_grid - y) / radius_y) ** 2 
            )
            
            mask = distance <= 1
            
            brightness = np.random.choice([
                np.random.randint(70, 110),
                np.random.randint(145, 190)
            ])
            
            alpha = np.random.uniform(0.25, 0.5)
            
            result[mask] = (
                result[mask] * (1 - alpha)
                + brightness * alpha
            )
            
        return np.clip(result, 0, 255).astype(np.uint8)


class NeonFilter(ImageFilter):
    def __init__(self, intensity=1.5, brightness=1.0):
        self.intensity = intensity
        self.brightness = brightness
        
    def apply_filter(self, image):
        gray = GrayscaleFilter().apply_filter(image).astype(np.float32)

        grad_x = np.abs(np.roll(gray, -1, axis=1) -
                        np.roll(gray, 1, axis=1))
        
        grad_y = np.abs(np.roll(gray, -1, axis=0) -
                        np.roll(gray, 1, axis=0))
        
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        glow = cv2.GaussianBlur(edges, (0, 0), sigmaX=8)
        
        dark = image.astype(np.float32) * 0.5
        
        neon = np.zeros_like(image, dtype=np.float32)
        
        neon[:, :, 0] = glow * 1.2
        neon[:, :, 1] = glow * 0.3
        neon[:, :, 2] = glow * 1.2
        
        result = dark + neon * self.intensity
        result *= self.brightness
        
        return np.clip(result, 0, 255).astype(np.uint8)