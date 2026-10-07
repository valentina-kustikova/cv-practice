import numpy as np


def check_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] == 0 or image.shape[1] == 0):
        raise ValueError("Ожидается непустое изображение uint8 с тремя каналами BGR")
    return image.astype(np.float32)


def check_range(name, value, minimum, maximum):
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}: допустимый диапазон [{minimum}, {maximum}]")


def to_uint8(image):
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8)


def brightness(image):
    return 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]


def resize(image, width, height):
    check_image(image)
    if (not isinstance(width, (int, np.integer))
            or not isinstance(height, (int, np.integer))
            or width < 1 or height < 1):
        raise ValueError("Ширина и высота должны быть положительными целыми числами")
    old_height, old_width = image.shape[:2]
    columns = np.arange(width) * old_width // width
    rows = np.arange(height) * old_height // height
    return image[rows[:, None], columns[None, :]].copy()


def grayscale(image):
    gray = brightness(check_image(image))
    return to_uint8(np.repeat(gray[:, :, None], 3, axis=2))


def antique(image, strength=1.0):
    check_range("strength", strength, 0, 1)
    source = check_image(image)
    blue, green, red = source[:, :, 0], source[:, :, 1], source[:, :, 2]
    sepia = np.stack((
        0.131 * blue + 0.534 * green + 0.272 * red,
        0.168 * blue + 0.686 * green + 0.349 * red,
        0.189 * blue + 0.769 * green + 0.393 * red,
    ), axis=2)
    return to_uint8((1 - strength) * source + strength * np.clip(sepia, 0, 255))


def fade(image, strength=0.5):
    check_range("strength", strength, 0, 1)
    source = check_image(image)
    gray = brightness(source)[:, :, None]
    faded = (1 - strength) * source + strength * gray
    lift = 64 * strength
    return to_uint8(lift + faded * (255 - lift) / 255)


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
    distance = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
    white = np.clip((distance - (1 - softness)) / softness, 0, 1)
    white = white[:, :, None]
    return to_uint8(source * (1 - white) + 255 * white)


def scratches(image, count=12, noise=10.0, seed=None):
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count должен быть целым неотрицательным числом")
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed должен быть целым неотрицательным числом")
    check_range("noise", noise, 0, 255)
    source = check_image(image)
    height, width = source.shape[:2]
    random = np.random.default_rng(seed)
    result = source + random.normal(0, noise, (height, width, 1))
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
        horizontal += weight * padded[:, index:index + width]
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(image)
    for index, weight in enumerate(weights):
        result += weight * padded[index:index + height, :]
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
    magnitude = np.hypot(right - left, bottom - top) / 4
    edges = np.where(magnitude > threshold, np.clip(magnitude / 255, 0, 1), 0)
    light = edges + glow * blur(edges, radius)
    color = np.array([255, 180, 40], dtype=np.float32)
    return to_uint8(0.1 * source + light[:, :, None] * color)
