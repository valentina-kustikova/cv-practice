import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Яркость Y = 0.299R + 0.587G + 0.114B."""
    img = image.astype(np.float32)
    return 0.299 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Приближение гауссова размытия: три прохода усреднения в окне 2r+1."""
    img = image.astype(np.float32)
    r = max(1, round(sigma))
    k = 2 * r + 1
    for _ in range(3):
        for axis in (0, 1):
            pad = [(0, 0)] * img.ndim
            pad[axis] = (r + 1, r)
            s = np.cumsum(np.pad(img, pad, mode="edge"), axis=axis)
            n = img.shape[axis]
            img = (np.take(s, np.arange(k, k + n), axis=axis) - np.take(s, np.arange(n), axis=axis)) / k
    return img


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Наложение «Экран»."""
    return 255 - (255 - a) * (255 - b) / 255
