import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Filter:    
    def __init__(self, name):
        self.name = name
    
    def create_filter(filter_type, **kwargs):
        filter_map = {
            'resize': ResizeImage,
            'sepia': SepiaFilter,
            'vinetka': VignetteFilter,
            'pixelize': PixelateFilter,
            'rect_frame': RectBorder,
            'frame': FrameFilter,
            'bliki': BliksFilter,
            'watercolor': WatercolorFilter
        }
        
        if filter_type not in filter_map:
            raise ValueError(f"Неизвестный тип фильтра: {filter_type}")
        
        return filter_map[filter_type](**kwargs)


class ResizeImage(Filter):    
    def __init__(self):
        super().__init__("Изменение размера")

    def apply(self, image, parameters):
        scale_percent = parameters.get('scale')
        height = parameters.get('height')
        width = parameters.get('width')
        org_height, org_width = image.shape[:2]
        if scale_percent is not None:
            n_height = int(org_height * scale_percent / 100)
            n_width = int(org_width * scale_percent / 100)
        elif height is not None and width is not None:
            n_height = height
            n_width = width
        else:
            logger.info("Параметры изменения размера не предоставлены, возвращается исходное изображение")
            return image.copy()
        if scale_percent is not None and scale_percent < 100:
            interpolation = cv.INTER_AREA  
        elif scale_percent is not None and scale_percent > 100:
            interpolation = cv.INTER_CUBIC  
        else:
            interpolation = cv.INTER_LINEAR 
            
        logger.info(f"Изменение размера изображения с {org_width}x{org_height} на {n_width}x{n_height}")
        return cv.resize(image, (n_width, n_height), interpolation=interpolation)


class SepiaFilter(Filter):    
    def __init__(self):
        super().__init__("Сепия")

    def apply(self, image, parameters):
        intensity = parameters.get('intensity', 1.0)
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],  
            [0.349, 0.686, 0.168], 
            [0.393, 0.769, 0.189]  
        ])
        identity_matrix = np.eye(3)
        if intensity <= 1.0:
            adjusted_matrix = intensity * sepia_matrix + (1 - intensity) * identity_matrix
        else:
            adjusted_matrix = sepia_matrix * intensity
            row_sums = adjusted_matrix.sum(axis=1, keepdims=True)
            adjusted_matrix = adjusted_matrix / np.maximum(row_sums, 1.0)

        logger.info(f"Применение фильтра сепии с интенсивностью {intensity}")
        sepia_image = cv.transform(image, adjusted_matrix)
        return np.clip(sepia_image, 0, 255).astype(np.uint8)


class VignetteFilter(Filter):    
    def __init__(self):
        super().__init__("Виньетка")

    def _create_vignette_mask(self, height, width, strength):
        center_x = width // 2
        center_y = height // 2
        x, y = np.ogrid[:height, :width]
        x_new = (x - center_x)**2 / (2 * (width * strength)**2)
        y_new = (y - center_y)**2 / (2 * (height * strength)**2)
        mask = np.exp(-(x_new + y_new))
        return mask / mask.max()
    
    def apply(self, image, parameters):
        strength = parameters.get('strength', 0.5)
        logger.info(f"Применение фильтра виньетки с силой {strength}")
        mask = self._create_vignette_mask(image.shape[0], image.shape[1], strength)
        result = image.astype(np.float32)
        result = result * mask[:, :, np.newaxis]  
        return np.clip(result, 0, 255).astype(np.uint8)


class PixelateFilter(Filter):   
    def __init__(self):
        super().__init__("Пикселизация")

    def apply(self, image, parameters):
        block_size = parameters.get('block_size', 10)
        x = parameters.get('x', 0)
        y = parameters.get('y', 0)
        width = parameters.get('width')
        height = parameters.get('height')
        result = image.copy()
        if width is None:
            width = image.shape[1] - x
        if height is None:
            height = image.shape[0] - y
        x = max(0, x)
        y = max(0, y)
        x_end = min(x + width, image.shape[1])
        y_end = min(y + height, image.shape[0])
        if x_end <= x or y_end <= y:
            logger.warning("Указана неверная область для пикселизации")
            return result
        actual_width = x_end - x
        actual_height = y_end - y
        area = result[y:y_end, x:x_end]
        block_size = max(2, min(block_size, min(actual_width, actual_height) // 2))
        small_width = max(1, actual_width // block_size)
        small_height = max(1, actual_height // block_size)
        small_area = cv.resize(area, (small_width, small_height), interpolation=cv.INTER_LINEAR)
        pixelated = cv.resize(small_area, (actual_width, actual_height), interpolation=cv.INTER_NEAREST)
        result[y:y_end, x:x_end] = pixelated
        logger.info(f"Пикселизирована область: x={x}, y={y}, ширина={actual_width}, высота={actual_height}, размер блока={block_size}")
        return result


class RectBorder(Filter):    
    def __init__(self):
        super().__init__("Прямоугольная рамка")

    def apply(self, image, parameters):
        border_width = parameters.get('border_width', 10)
        color = parameters.get('color', (255, 255, 0))
        result = image.copy()
        img_height, img_width = image.shape[:2]
        border_width = min(border_width, img_height // 2, img_width // 2)
        result[:border_width, :] = color   
        result[-border_width:, :] = color  
        result[:, :border_width] = color   
        result[:, -border_width:] = color  
        logger.info(f"Применена прямоугольная рамка с шириной {border_width} и цветом {color}")
        return result

#
class FrameFilter(Filter):    
    def __init__(self):
        super().__init__("Фигурная одноцветная рамка")

    def apply(self, image, parameters):
        frame_type = parameters.get('frame_type', 'circle')
        border_width = parameters.get('border_width', 10)
        color = parameters.get('color', (255, 255, 0))
        result = image.copy()
        img_height, img_width = image.shape[:2]
        if frame_type == "circle":
            center = (img_width // 2, img_height // 2)
            radius = min(img_height, img_width) // 2 - border_width // 2
            cv.circle(result, center, radius, color, border_width)
        elif frame_type == "diamond":
            points = np.array([
                [img_width // 2, 0],           
                [img_width, img_height // 2],  
                [img_width // 2, img_height],  
                [0, img_height // 2]           
            ], dtype=np.int32)
            cv.polylines(result, [points], color=color, thickness=border_width, isClosed=True)
        else: 
            result[:border_width, :] = color
            result[-border_width:, :] = color
            result[:, :border_width] = color
            result[:, -border_width:] = color
        logger.info(f"Применена {frame_type} рамка с шириной {border_width} и цветом {color}")
        return result


class BliksFilter(Filter):    
    def __init__(self):
        super().__init__("Блики")

    def apply(self, image, parameters):
        intensity = parameters.get('intensity', 0.5)
        cx = parameters.get('cx')
        cy = parameters.get('cy')
        img_height, img_width = image.shape[:2]
        if cx is None:
            cx = img_width // 2
        if cy is None:
            cy = img_height // 2
        y, x = np.ogrid[:img_height, :img_width]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        flare = np.exp(-2 * distance / max_dist) * intensity
        flare = np.dstack([flare, flare, flare])  
        result = image.astype(np.float32) + flare * 255
        logger.info(f"Применены блики с центром ({cx}, {cy}) и интенсивностью {intensity}")
        return np.clip(result, 0, 255).astype(np.uint8)


class WatercolorFilter(Filter):
    def __init__(self):
        super().__init__("Текстура акварельной бумаги")
        
    def apply(self, image, parameters):
        intensity = parameters.get('intensity', 0.2)
        height, width = image.shape[:2]
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        texture = np.zeros_like(image, dtype=np.uint8)
        for i in range(3):
            texture[:, :, i] = cv.GaussianBlur(noise[:, :, i], (25, 25), 0)
        logger.info(f"Применен акварельный фильтр с интенсивностью {intensity}")
        return cv.addWeighted(image, 1 - intensity, texture, intensity, 0)
    
    