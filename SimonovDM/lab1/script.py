import argparse
import cv2
import numpy as np
import abc
import logging

class Filter(abc.ABC):
    @abc.abstractmethod
    def apply_filter(self, image, **kwargs):
        pass

    @staticmethod
    def create_filter(name, **kwargs):
        if name == "resize":
            return ResizeFilter(**kwargs)
        elif name == "sepia":
            return SepiaFilter(**kwargs)
        elif name == "vignette":
            return VignetteFilter(**kwargs)
        elif name == "pixelation":
            return PixelationFilter(**kwargs)
        elif name == "frame":
            return FrameFilter(**kwargs)
        elif name == "frame_figure":
            return FrameFigureFilter(**kwargs)
        elif name == "glare":
            return GlareFilter(**kwargs)
        elif name == "aqua_texture":
            return AquaTextureFilter(**kwargs)
        else:
            raise ValueError(f"Неизвестный фильтр: {name}")

class ResizeFilter(Filter):
    def __init__(self, width=None, height=None, scale=None):
        self.width = width
        self.height = height
        self.scale = scale

    def apply_filter(self, image, **kwargs):
        h, w = image.shape[:2]
        width = self.width or kwargs.get("width")
        height = self.height or kwargs.get("height")
        scale_factor = self.scale or kwargs.get("scale")

        if (width is None) and (height is None) and (scale_factor is None):
            return image.copy()
        if width == w and height == h:
            return image.copy()

        if scale_factor is not None:
            if scale_factor == 1:
                return image.copy()
            new_height = int(h * scale_factor)
            new_width = int(w * scale_factor)
        else:
            new_width = width if width is not None else w
            new_height = height if height is not None else h

        scale_h = float(h / new_height)
        scale_w = float(w / new_width)

        y = (np.arange(new_height) * scale_h).astype(int)
        x = (np.arange(new_width) * scale_w).astype(int)
        x_neigh_index, y_neigh_index = np.meshgrid(x, y)

        result = image[y_neigh_index, x_neigh_index]
        return result

class SepiaFilter:
    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def apply_filter(self, image, **kwargs):
        sepia_img = np.zeros_like(image, np.uint8)
        intensity = self.intensity or kwargs.get("intensity")

        sepia_img[:, :, 2] = np.clip(0.393 * intensity * image[:, :, 2] + 0.769 * intensity * image[:, :, 1] + 0.189 * intensity * image[:, :, 0], 0, 255)
        sepia_img[:, :, 1] = np.clip(0.349 * intensity * image[:, :, 2] + 0.686 * intensity * image[:, :, 1] + 0.168 * intensity * image[:, :, 0], 0, 255)
        sepia_img[:, :, 0] = np.clip(0.272 * intensity * image[:, :, 2] + 0.534 * intensity * image[:, :, 1] + 0.131 * intensity * image[:, :, 0], 0, 255)
        return sepia_img

class VignetteFilter(Filter):
    def __init__(self, intensity=0.8, radius=0.8):
        self.intensity = intensity
        self.radius = radius

    def vignette_mask(self, height, width, intensity, radius):
        y, x = np.ogrid[:height, :width]
        center_x = width // 2
        center_y = height // 2

        max_radius = min(center_x, center_y) * radius
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        mask = np.exp(-((distance / max_radius) ** 2) * intensity)
        mask = mask / np.max(mask)

        return mask

    def apply_filter(self, image, **kwargs):
        h, w = image.shape[:2]
        intensity = self.intensity or kwargs.get("intensity")
        radius = self.radius or kwargs.get("radius")

        mask = self.vignette_mask(h, w, intensity, radius)
        image = image * mask[:, :, np.newaxis]
        image = image.astype(np.uint8)
        return image

class PixelationFilter(Filter):
    def __init__(self, start_x, start_y, end_x, end_y, block_size = 2):
        self.rectangle_start = (start_x, start_y)
        self.rectangle_end = (end_x, end_y)
        self.block_size = block_size

    def apply_filter(self, image, **kwargs):
        rectangle_start = self.rectangle_start or kwargs.get("rectangle_start")
        rectangle_end = self.rectangle_end or kwargs.get("rectangle_end")
        block_size = self.block_size or kwargs.get("block_size")

        if rectangle_start is None:
            x1, y1 = 0, 0
        else:
            x1, y1 = rectangle_start

        if rectangle_end is None:
            x2, y2 = image.shape[1], image.shape[0]
        else:
            x2, y2 = rectangle_end


        result = image.copy()
        for i in range(y1, y2, block_size):
            for j in range(x1, x2, block_size):
                block_y2 = min(i + block_size, y2)
                block_x2 = min(j + block_size, x2)
                block = result[i:block_y2, j:block_x2]
                avg_color = np.mean(block, axis=(0, 1), dtype=int)
                result[i:block_y2, j:block_x2] = avg_color

        return result

class FrameFilter(Filter):
    def __init__(self, width=None, r=0, g=0, b=0):
        self.frame_width = width
        self.R, self.G, self.B = r, g, b

    def apply_filter(self, image, **kwargs):
        frame_width = self.frame_width or kwargs.get("width")
        R = self.R or kwargs.get("r",0)
        G = self.G or kwargs.get("g",0)
        B = self.B or kwargs.get("b",0)

        if frame_width is None:
            return image

        result = image.copy()
        h, w = image.shape[:2]

        result[0:frame_width] = [B, G, R]
        result[-frame_width:] = [B, G, R]
        result[frame_width:-frame_width, 0:frame_width] = [B, G, R]
        result[frame_width:-frame_width, -frame_width:] = [B, G, R]
        return result

class FrameFigureFilter(Filter):
    def __init__(self, frame_number = 1):
        self.frame_number = frame_number

    def apply_filter(self, image, **kwargs):
        frame_files = ["frame/frame1.png", "frame/frame2.jpeg"]
        frame_number = self.frame_number or kwargs.get("frame_number")

        if frame_number < 1 or frame_number > len(frame_files):
            logging.error(f"Ошибка: номер рамки должен быть от 1 до {len(frame_files)}")
            return image

        frame = read_image(frame_files[frame_number - 1])
        h, w = image.shape[:2]
        result = image.copy()
        if frame.shape != result.shape:
            frame = ResizeFilter(width=w, height=h).apply_filter(frame)

        for x in range(w):
            for y in range(h):
                if np.all(frame[y][x] <= [230, 230, 230]):
                    result[y][x] = frame[y][x]

        return result

class GlareFilter(Filter):
    def __init__(self, center_x = 0.5, center_y = 0.5, intensity = 1, scale = 1):
        self.flare_center = (center_x, center_y)
        self.intensity = intensity
        self.scale = scale

    def apply_filter(self, image, **kwargs):
        h_img, w_img = image.shape[:2]
        flare_center = self.flare_center or kwargs.get("flare_center")
        intensity = self.intensity or kwargs.get("intensity")
        scale = self.scale or kwargs.get("scale")

        glare = read_image("glare/glare1.png")

        new_w = int(glare.shape[1] * scale)
        new_h = int(glare.shape[0] * scale)
        glare = ResizeFilter(new_w, new_h).apply_filter(glare)

        cx = int(flare_center[0] * w_img)
        cy = int(flare_center[1] * h_img)

        x_start = cx - new_w // 2
        y_start = cy - new_h // 2
        x_end = x_start + new_w
        y_end = y_start + new_h

        img_x_start = max(0, x_start)
        img_y_start = max(0, y_start)
        img_x_end = min(w_img, x_end)
        img_y_end = min(h_img, y_end)

        glare_x_start = max(0, -x_start)
        glare_y_start = max(0, -y_start)
        glare_x_end = glare_x_start + (img_x_end - img_x_start)
        glare_y_end = glare_y_start + (img_y_end - img_y_start)

        result = image.copy()
        img_region = image[img_y_start:img_y_end, img_x_start:img_x_end].astype(np.float32)
        glare_region = glare[glare_y_start:glare_y_end, glare_x_start:glare_x_end].astype(np.float32)

        img_region = img_region + glare_region * intensity
        result[img_y_start:img_y_end, img_x_start:img_x_end] = np.clip(img_region, 0, 255).astype(np.uint8)
        return result

class AquaTextureFilter(Filter):
    def __init__(self, intensity=1.0):
        self.intensity = intensity

    def bgr_to_gray(self, image):
        if image.shape[2] < 3:
            raise ValueError("Ожидается цветное изображение с 3 каналами (BGR)")

        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]

        gray = 0.114 * B + 0.587 * G + 0.299 * R
        return gray

    def apply_filter(self, image, **kwargs):
        h, w = image.shape[:2]
        intensity = self.intensity or kwargs.get("intensity")

        aqua = read_image("aqua/aqua1.png")
        aqua = ResizeFilter(w, h).apply_filter(aqua)

        image_float = image.astype(np.float32) / 255.0
        aqua_float = aqua.astype(np.float32) / 255.0

        aqua_gray = self.bgr_to_gray(aqua_float)
        aqua_gray = aqua_gray[:, :, np.newaxis]

        result = image_float * (aqua_gray * intensity)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

def read_image(image_file):
    image = cv2.imread(image_file)
    if image is None:
        raise f"Ошибка: не удалось загрузить изображение {image_file}"
    return image

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        type=str,
                        required=True,
                        help='Путь к изображению')

    subparsers = parser.add_subparsers(dest='filter_type',
                                       required=True,
                                       help='Тип фильтра',)

    resize = subparsers.add_parser('resize',
                                   help='Изменение размера')
    resize.add_argument('--width',
                        type=int,
                        help='Новая ширина')
    resize.add_argument('--height',
                        type=int,
                        help='Новая высота')
    resize.add_argument('--scale',
                        type=float,
                        help='Коэффициент масштабирования')

    sepia = subparsers.add_parser('sepia',
                                  help='Применение фотоэффекта сепии')
    sepia.add_argument('--intensity',
                       type=float,
                       default=1.0,
                       help='Интенсивность применения сепии')

    vignette = subparsers.add_parser('vignette',
                                     help='Применение фотоэффекта виньетки')
    vignette.add_argument('--intensity',
                          type=float,
                          default=0.8,
                          help='Интенсивность виньетки')
    vignette.add_argument('--radius',
                          type=float,
                          default=0.8,
                          help='Радиус виньетки')

    pixelation = subparsers.add_parser('pixelation',
                                       help='Пикселизация области')
    pixelation.add_argument('--start_x',
                            type=int,
                            required=True,
                            help='X-координата начала')
    pixelation.add_argument('--start_y',
                            type=int,
                            required=True,
                            help='Y-координата начала')
    pixelation.add_argument('--end_x',
                            type=int,
                            required=True,
                            help='X-координата конца')
    pixelation.add_argument('--end_y',
                            type=int,
                            required=True,
                            help='Y-координата конца')
    pixelation.add_argument('--block_size',
                            type=int,
                            default=10,
                            help='Размер блока')

    pix = subparsers.add_parser('pix',
                                       help='Пикселизация области')
    pix.add_argument('--start_x',
                            type=int,
                            required=True,
                            help='X-координата начала')
    pix.add_argument('--start_y',
                            type=int,
                            required=True,
                            help='Y-координата начала')
    pix.add_argument('--end_x',
                            type=int,
                            required=True,
                            help='X-координата конца')
    pix.add_argument('--end_y',
                            type=int,
                            required=True,
                            help='Y-координата конца')
    pix.add_argument('--block_size',
                            type=int,
                            default=10,
                            help='Размер блока')

    frame = subparsers.add_parser('frame',
                                  help='Прямоугольная рамка')
    frame.add_argument('--width',
                       type=int,
                       default=None,
                       help='Ширина рамки')
    frame.add_argument('--r',
                       type=int,
                       default=0,
                       help='Красный цвет')
    frame.add_argument('--g',
                       type=int,
                       default=0,
                       help='Зелёный цвет')
    frame.add_argument('--b',
                       type=int,
                       default=0,
                       help='Синий цвет')

    frame_figure = subparsers.add_parser('frame_figure',
                                         help='Фигурная рамка')
    frame_figure.add_argument('--frame_number',
                              type=int,
                              default=1,
                              help='Номер рамки')
    ff = subparsers.add_parser('ff',
                                         help='Фигурная рамка')
    ff.add_argument('--frame_number',
                              type=int,
                              default=1,
                              help='Номер рамки')

    glare = subparsers.add_parser('glare',
                                  help='Блики на изображении')
    glare.add_argument('--center_x',
                       type=float,
                       default=0.5,
                       help='X-координата центра блика (0.0-1.0)')
    glare.add_argument('--center_y',
                       type=float,
                       default=0.5,
                       help='Y-координата центра блика (0.0-1.0)')
    glare.add_argument('--intensity',
                       type=float,
                       default=1.0,
                       help='Интенсивность блика')
    glare.add_argument('--scale',
                       type=float,
                       default=1.0,
                       help='Коэффициент масштабирования')

    aqua_texture = subparsers.add_parser('aqua_texture',
                                         help='Текстура акварели')
    aqua_texture.add_argument('--intensity',
                              type=float,
                              default=0.5,
                              help='Интенсивность текстуры')

    aqt = subparsers.add_parser('aqt',
                                help='Текстура акварели')
    aqt.add_argument('--intensity',
                     type=float,
                     default=0.5,
                     help='Интенсивность текстуры')

    return parser.parse_args()
