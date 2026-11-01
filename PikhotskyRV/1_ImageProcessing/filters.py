from abc import ABC, abstractmethod

import numpy as np


def check_image(image):
    error = "Ожидается непустое изображение uint8 с тремя каналами BGR"
    if not isinstance(image, np.ndarray):
        raise ValueError(error)
    if image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError(error)
    height, width, channels = image.shape
    if height == 0 or width == 0 or channels != 3:
        raise ValueError(error)
    return image.astype(np.float32)


def check_range(name, value, minimum, maximum):
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: допустимый диапазон [{minimum}, {maximum}]")


def to_uint8(image):
    image = np.clip(image, 0, 255)
    image = np.rint(image)
    return image.astype(np.uint8)


def brightness(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    return 0.114 * blue + 0.587 * green + 0.299 * red


def resize(image, width, height):
    check_image(image)
    if (not isinstance(width, (int, np.integer))
            or not isinstance(height, (int, np.integer))
            or width < 1 or height < 1):
        raise ValueError("Ширина и высота должны быть положительными целыми числами")
    old_height, old_width = image.shape[:2]
    columns = np.arange(width) * old_width // width
    result = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        source_row = row * old_height // height
        result[row] = image[source_row, columns]
    return result


def grayscale(image):
    source = check_image(image)
    gray = brightness(source)
    result = np.zeros_like(source)
    for channel in range(3):
        result[:, :, channel] = gray
    return to_uint8(result)


def antique(image, strength=1.0):
    check_range("strength", strength, 0, 1)
    source = check_image(image)
    blue = source[:, :, 0]
    green = source[:, :, 1]
    red = source[:, :, 2]
    sepia = np.zeros_like(source)
    sepia[:, :, 0] = 0.131 * blue + 0.534 * green + 0.272 * red
    sepia[:, :, 1] = 0.168 * blue + 0.686 * green + 0.349 * red
    sepia[:, :, 2] = 0.189 * blue + 0.769 * green + 0.393 * red
    sepia = np.clip(sepia, 0, 255)
    result = (1 - strength) * source + strength * sepia
    return to_uint8(result)


def fade(image, strength=0.5):
    check_range("strength", strength, 0, 1)
    source = check_image(image)
    gray = brightness(source)
    lift = 64 * strength
    result = np.zeros(source.shape, dtype=np.float64)
    for channel in range(3):
        faded = (1 - strength) * source[:, :, channel] + strength * gray
        result[:, :, channel] = lift + faded * (255 - lift) / 255
    return to_uint8(result)


def film(image, gamma=1.0):
    check_range("gamma", gamma, 0.1, 5)
    source = check_image(image) / 255
    return to_uint8(255 * (1 - source) ** gamma)


def matte(image, softness=0.3):
    check_range("softness", softness, 0.01, 1)
    source = check_image(image)
    height, width = source.shape[:2]
    x = (2 * np.arange(width) - (width - 1)) / max(width - 1, 1)
    y = (2 * np.arange(height) - (height - 1)) / max(height - 1, 1)
    x_squared = x ** 2
    y_squared = y ** 2
    distance = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        distance[row] = np.sqrt(x_squared + y_squared[row])
    white = np.clip((distance - (1 - softness)) / softness, 0, 1)
    result = np.zeros(source.shape, dtype=np.float64)
    for channel in range(3):
        result[:, :, channel] = source[:, :, channel] * (1 - white) + 255 * white
    return to_uint8(result)


def scratches(image, count=12, noise=10.0, seed=None):
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count должен быть целым неотрицательным числом")
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed должен быть целым неотрицательным числом")
    check_range("noise", noise, 0, 255)
    source = check_image(image)
    height, width = source.shape[:2]
    random = np.random.default_rng(seed)
    grain = random.normal(0, noise, (height, width))
    result = np.zeros(source.shape, dtype=np.float64)
    for channel in range(3):
        result[:, :, channel] = source[:, :, channel] + grain
    for _ in range(count):
        column = random.integers(0, width)
        start = random.integers(0, height)
        end = random.integers(start + 1, height + 1)
        shade = random.choice([0, 255])
        result[start:end, column, :] = shade
    return to_uint8(result)


def blur(image, radius):
    if radius == 0:
        return image.copy()
    offsets = np.arange(-radius, radius + 1)
    sigma = max(radius / 2, 0.5)
    weights = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()
    height, width = image.shape
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.zeros_like(image)
    for index, weight in enumerate(weights):
        neighbors = padded[:, index:index + width]
        horizontal += weight * neighbors
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(image)
    for index, weight in enumerate(weights):
        neighbors = padded[index:index + height, :]
        result += weight * neighbors
    return result


def neon(image, threshold=25.0, radius=4, glow=2.0):
    check_range("threshold", threshold, 0, 255)
    check_range("glow", glow, 0, 10)
    if not isinstance(radius, (int, np.integer)) or not 0 <= radius <= 50:
        raise ValueError("radius должен быть целым числом от 0 до 50")
    source = check_image(image)
    gray = np.pad(brightness(source), 1, mode="edge")
    left = gray[:-2, :-2] + 2 * gray[1:-1, :-2] + gray[2:, :-2]
    right = gray[:-2, 2:] + 2 * gray[1:-1, 2:] + gray[2:, 2:]
    top = gray[:-2, :-2] + 2 * gray[:-2, 1:-1] + gray[:-2, 2:]
    bottom = gray[2:, :-2] + 2 * gray[2:, 1:-1] + gray[2:, 2:]
    gradient_x = right - left
    gradient_y = bottom - top
    magnitude = np.hypot(gradient_x, gradient_y) / 4
    edges = np.clip(magnitude / 255, 0, 1)
    edges[magnitude <= threshold] = 0
    light = edges + glow * blur(edges, radius)
    color = [255, 180, 40]
    result = np.zeros(source.shape, dtype=np.float64)
    for channel in range(3):
        result[:, :, channel] = 0.1 * source[:, :, channel] + light * color[channel]
    return to_uint8(result)


class ImageFilter(ABC):
    @staticmethod
    def get_filter(name, **parameters):
        if name not in FILTER_CLASSES:
            raise ValueError(f"Неизвестный фильтр: {name}")
        filter_class = FILTER_CLASSES[name]
        return filter_class(**parameters)

    @abstractmethod
    def apply_filter(self, image):
        pass


class Resize(ImageFilter):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def apply_filter(self, image):
        return resize(image, self.width, self.height)


class RGB2GrayScale(ImageFilter):
    def apply_filter(self, image):
        return grayscale(image)


class Antique(ImageFilter):
    def __init__(self, strength=1.0):
        self.strength = strength

    def apply_filter(self, image):
        return antique(image, self.strength)


class FadeColor(ImageFilter):
    def __init__(self, strength=0.5):
        self.strength = strength

    def apply_filter(self, image):
        return fade(image, self.strength)


class Film(ImageFilter):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def apply_filter(self, image):
        return film(image, self.gamma)


class Matte(ImageFilter):
    def __init__(self, softness=0.3):
        self.softness = softness

    def apply_filter(self, image):
        return matte(image, self.softness)


class Scratches(ImageFilter):
    def __init__(self, count=12, noise=10.0, seed=None):
        self.count = count
        self.noise = noise
        self.seed = seed

    def apply_filter(self, image):
        return scratches(image, self.count, self.noise, self.seed)


class Neon(ImageFilter):
    def __init__(self, threshold=25.0, radius=4, glow=2.0):
        self.threshold = threshold
        self.radius = radius
        self.glow = glow

    def apply_filter(self, image):
        return neon(image, self.threshold, self.radius, self.glow)


FILTER_CLASSES = {
    "resize": Resize,
    "grayscale": RGB2GrayScale,
    "antique": Antique,
    "fade": FadeColor,
    "film": Film,
    "matte": Matte,
    "scratches": Scratches,
    "neon": Neon,
}
