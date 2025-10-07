import numpy as np

# Изменение разрешения изображения
def resize_image(img, scale=0.5):
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)
    resized = img[y_idx][:, x_idx]
    return resized


# Сепия
def apply_sepia(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = img @ sepia_filter.T
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
    return sepia_img


# Виньетка
def apply_vignette(img, strength=0.5):
    rows, cols = img.shape[:2]
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X ** 2 + Y ** 2)
    mask = np.exp(-radius ** 2 / (2 * strength ** 2))
    vignette = img * mask[..., np.newaxis]
    return np.clip(vignette, 0, 255).astype(np.uint8)


# Пикселизация области
def pixelate_region(img, x1, y1, x2, y2, pixel_size=10):
    img_copy = img.copy()
    region = img_copy[y1:y2, x1:x2]
    h, w = region.shape[:2]
    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = region[y:y + pixel_size, x:x + pixel_size]
            color = block.mean(axis=(0, 1))
            region[y:y + pixel_size, x:x + pixel_size] = color
    img_copy[y1:y2, x1:x2] = region
    return img_copy


# Прямоугольная рамка
def add_rectangular_frame(img, color=(0, 0, 255), thickness=20):
    framed = img.copy()
    framed[:thickness, :] = color
    framed[-thickness:, :] = color
    framed[:, :thickness] = color
    framed[:, -thickness:] = color
    return framed


# Фигурная рамка
def add_figure_frame(img, color=(0, 255, 0), thickness=20, frame_type='wave'):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x = np.arange(w)
    if frame_type == 'wave':
        y = (np.sin(x / 20) * 10 + thickness).astype(np.int32)
        for i in range(w):
            mask[:y[i], i] = 1
            mask[-y[i]:, i] = 1
    elif frame_type == 'triangle':
        for i in range(thickness):
            mask[i::thickness * 2, :] = 1
            mask[-i - 1::thickness * 2, :] = 1
    mask[:, :thickness] = 1
    mask[:, -thickness:] = 1
    frame = img.copy()
    frame[mask > 0] = color
    return frame


# Эффект бликов
def add_lens_flare(img, center=None, intensity=0.7):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    flare = np.exp(-(dist / (0.5 * w)) ** 2)
    flare = flare[..., np.newaxis]
    flare_color = np.array([1.0, 0.8, 0.6])
    flare_img = img.astype(np.float32) / 255.0
    flare_img += flare * flare_color * intensity
    return np.clip(flare_img * 255, 0, 255).astype(np.uint8)


# Текстура акварельной бумаги
def add_paper_texture(img, scale=5, intensity=0.2):
    noise = np.random.normal(0.5, 0.2, img.shape[:2])
    noise = np.clip(noise, 0, 1)

    # Простое размытие усреднением
    kernel_size = max(1, scale)
    pad = kernel_size // 2
    padded = np.pad(noise, pad, mode='reflect')
    smoothed = np.zeros_like(noise)
    for y in range(noise.shape[0]):
        for x in range(noise.shape[1]):
            region = padded[y:y + kernel_size, x:x + kernel_size]
            smoothed[y, x] = np.mean(region)

    texture = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    texture = np.repeat(texture[:, :, np.newaxis], 3, axis=2)
    img_textured = img.astype(np.float32) / 255.0
    img_textured *= (1 - intensity + texture * intensity)
    return np.clip(img_textured * 255, 0, 255).astype(np.uint8)
