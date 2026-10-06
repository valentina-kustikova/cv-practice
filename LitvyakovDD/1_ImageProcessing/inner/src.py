import numpy as np


def clip_image(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def to_gray(image):
    image = image.astype(np.float32)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    return 0.114 * blue + 0.587 * green + 0.299 * red


def check_image(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Нужно цветное изображение.")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("У изображения неправильный размер.")

def resize(image, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Ширина и высота должны быть больше нуля.")

    image_height, image_width = image.shape[:2]
    x_indices = (np.arange(width) * image_width / width).astype(int)
    y_indices = (np.arange(height) * image_height / height).astype(int)
    return image[y_indices[:, None], x_indices]


def grayscale(image):
    return clip_image(to_gray(image))

def antique(image, strength=1.0):
    image = image.astype(np.float32)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    sepia = np.stack((
        0.272 * red + 0.534 * green + 0.131 * blue,
        0.349 * red + 0.686 * green + 0.168 * blue,
        0.393 * red + 0.769 * green + 0.189 * blue
    ), axis=2)
    return clip_image(image * (1 - strength) + sepia * strength)

def fade(image, strength=1.0):
    original = image.astype(np.float32)
    gray = to_gray(image)[:, :, None]
    faded = original * (1 - 0.55 * strength) + gray * (0.55 * strength)
    faded = faded * (1 - 0.18 * strength) + 245 * (0.18 * strength)
    return clip_image(faded)


def film(image, strength=1.0):
    image = image.astype(np.float32)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    infrared = np.stack((
        0.65 * blue + 0.35 * red,
        0.75 * red + 0.25 * green,
        1.35 * green - 0.25 * blue + 20
    ), axis=2)
    return clip_image(image * (1 - strength) + infrared * strength)


def matte(image, strength=0.8):
    height, width = image.shape[:2]
    x_coordinates = (
        np.arange(width, dtype=np.float32) + 0.5 - width / 2
    ) / (width * 0.48)
    y_coordinates = (
        np.arange(height, dtype=np.float32) + 0.5 - height / 2
    ) / (height * 0.48)
    distance_from_center = np.sqrt(
        y_coordinates[:, None] ** 2 + x_coordinates[None, :] ** 2
    )
    white_amount = np.clip(
        (distance_from_center - 0.68) / 0.32, 0, 1
    )[:, :, None] * strength
    result = image.astype(np.float32) * (1 - white_amount) + 255 * white_amount
    return clip_image(result)


def scratches(image, count):
    random_generator = np.random.default_rng()
    result = image.astype(np.float32)
    height, width = image.shape[:2]
    for _ in range(count):
        column = random_generator.integers(0, width)
        start_row = random_generator.integers(0, height)
        end_row = random_generator.integers(start_row + 1, height + 1)
        scratch_color = 255 if random_generator.random() < 0.5 else 0
        result[start_row:end_row, column, :] = scratch_color

    return clip_image(result)


def neon(image, threshold=70.0, strength=0.7, brightness=0.40):
    """Сравнивает яркость каждого пикселя с соседями справа и снизу.
    Если сумма разниц больше threshold, пиксель считается контуром.
    Изображение затемняется до brightness, а цвет контура смешивается с (255, 45, 210)."""
    gray = to_gray(image)
    vertical_change = np.abs(gray[1:, :] - gray[:-1, :])
    horizontal_change = np.abs(gray[:, 1:] - gray[:, :-1])
    vertical_change = np.pad(vertical_change, ((0, 1), (0, 0)))
    horizontal_change = np.pad(horizontal_change, ((0, 0), (0, 1)))
    edges = vertical_change + horizontal_change > threshold

    result = image.astype(np.float32) * brightness
    neon_color = np.array([255, 45, 210], dtype=np.float32)
    result[edges] = (
        result[edges] * (1 - strength) + neon_color * strength
    )
    return clip_image(result)


FILTERS = {
    "resize": resize,
    "grayscale": grayscale,
    "antique": antique,
    "fade": fade,
    "film": film,
    "matte": matte,
    "scratches": scratches,
    "neon": neon,
}
