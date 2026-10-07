import cv2
import numpy as np


def resize_image(image, scale=0.5):
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            old_y = int(y / scale)
            old_x = int(x / scale)
            result[y, x] = image[old_y, old_x]

    return result


def gray_image(image):
    result = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            b, g, r = image[y, x]
            gray = int(0.114 * b + 0.587 * g + 0.299 * r)
            result[y, x] = [gray, gray, gray]

    return result


def antique_image(image):
    result = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            b, g, r = image[y, x]
            new_r = min(255, int(0.393 * r + 0.769 * g + 0.189 * b))
            new_g = min(255, int(0.349 * r + 0.686 * g + 0.168 * b))
            new_b = min(255, int(0.272 * r + 0.534 * g + 0.131 * b))
            result[y, x] = [new_b, new_g, new_r]

    return result


def fade_image(image):
    result = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            b, g, r = image[y, x]
            gray = int(0.114 * b + 0.587 * g + 0.299 * r)
            new_b = int(b * 0.5 + gray * 0.3 + 255 * 0.2)
            new_g = int(g * 0.5 + gray * 0.3 + 255 * 0.2)
            new_r = int(r * 0.5 + gray * 0.3 + 255 * 0.2)
            result[y, x] = [new_b, new_g, new_r]

    return result


def film_image(image):
    result = image.copy()

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            b, g, r = image[y, x]
            new_b = min(255, int(b * 1.2))
            new_g = min(255, int(g * 0.9))
            new_r = min(255, int(255 - r * 0.5))
            result[y, x] = [new_b, new_g, new_r]

    return result


def matte_image(image):
    result = image.copy()
    height, width = image.shape[:2]
    center_x = width / 2
    center_y = height / 2
    radius_x = width / 2
    radius_y = height / 2

    for y in range(height):
        for x in range(width):
            distance = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2

            if distance > 0.55:
                k = min(1, (distance - 0.55) / 0.45)
                b, g, r = image[y, x]
                result[y, x] = [
                    int(b * (1 - k) + 255 * k),
                    int(g * (1 - k) + 255 * k),
                    int(r * (1 - k) + 255 * k),
                ]

    return result


def scratches_image(image):
    result = antique_image(image)
    height, width = image.shape[:2]

    noise = np.random.randint(-25, 26, image.shape)
    result = np.clip(result.astype(int) + noise, 0, 255).astype(np.uint8)

    for _ in range(25):
        x = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        y2 = min(height - 1, y1 + np.random.randint(30, 120))
        cv2.line(result, (x, y1), (x, y2), (230, 230, 230), 1)

    return result


def neon_image(image):
    gray = gray_image(image)
    result = np.zeros_like(image)
    height, width = image.shape[:2]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pixel = int(gray[y, x][0])
            right = int(gray[y, x + 1][0])
            bottom = int(gray[y + 1, x][0])
            difference = abs(pixel - right) + abs(pixel - bottom)

            if difference > 50:
                result[y, x] = [255, 0, 255]
            else:
                result[y, x] = image[y, x] * 0.25

    return result


def apply_filter(image, filter_name):
    if filter_name == "resize":
        return resize_image(image)
    if filter_name == "gray":
        return gray_image(image)
    if filter_name == "antique":
        return antique_image(image)
    if filter_name == "fade":
        return fade_image(image)
    if filter_name == "film":
        return film_image(image)
    if filter_name == "matte":
        return matte_image(image)
    if filter_name == "scratches":
        return scratches_image(image)
    if filter_name == "neon":
        return neon_image(image)

    raise ValueError("Unknown filter")


FILTERS = ["resize", "gray", "antique", "fade", "film", "matte", "scratches", "neon"]
