import numpy as np


def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def to_gray(image):
    #каналы OpenCV B, G, R
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def blur(image, radius):
    img = image.astype(np.float32)
    size = 2 * radius + 1
    for _ in range(2):
        padded = np.pad(img, ((radius + 1, radius), (0, 0), (0, 0)), mode="edge")
        sums = np.cumsum(padded, axis=0)
        img = (sums[size:] - sums[:-size]) / size

        padded = np.pad(img, ((0, 0), (radius + 1, radius), (0, 0)), mode="edge")
        sums = np.cumsum(padded, axis=1)
        img = (sums[:, size:] - sums[:, :-size]) / size
    return img

def convolve(gray, kernel):
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge")
    result = np.zeros((h, w), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            result += kernel[i][j] * padded[i:i + h, j:j + w]
    return result
