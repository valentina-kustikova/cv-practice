
import numpy as np
import cv2

def resize_image(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    if height is None:
        ratio = width / w
        height = int(h * ratio)
    resized = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    for i in range(height):
        for j in range(width):
            src_i = min(int(i * h / height), h - 1)
            src_j = min(int(j * w / width), w - 1)
            resized[i, j] = image[src_i, src_j]
    return resized

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j].astype(np.float32)
            new_pixel = sepia_filter @ pixel[:3]
            result[i, j] = np.clip(new_pixel, 0, 255)
    return result.astype(np.uint8)

def apply_vignette(image, strength=0.6):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    dist = (x - cx)**2 + (y - cy)**2
    max_dist = (cx**2 + cy**2)
    mask = 1 - (dist / max_dist)
    mask = np.clip(mask, 0, 1)
    mask = 1 - strength + strength * mask
    result = image.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= mask
    return np.clip(result, 0, 255).astype(np.uint8)

def pixelize_region(image, x1, y1, x2, y2, pixel_size=10):
    img = image.copy()
    for i in range(y1, y2, pixel_size):
        for j in range(x1, x2, pixel_size):
            h_end = min(i + pixel_size, y2)
            w_end = min(j + pixel_size, x2)
            block = img[i:h_end, j:w_end]
            avg = np.mean(block, axis=(0, 1)).astype(np.uint8)
            img[i:h_end, j:w_end] = avg
    return img

def apply_solid_border(image, border_width, color=(255, 255, 255)):
    h, w, c = image.shape
    new_h, new_w = h + 2*border_width, w + 2*border_width
    result = np.full((new_h, new_w, c), color, dtype=image.dtype)
    result[border_width:border_width+h, border_width:border_width+w] = image
    return result

def apply_custom_border(image, border_width, border_type='dashed', color=(255, 255, 255)):
    img = image.copy()
    h, w = img.shape[:2]
    color = np.array(color, dtype=img.dtype)
    if border_type == 'dashed':
        dash = border_width * 2
        gap = border_width * 2
        x = 0
        while x < w:
            end = min(x + dash, w)
            img[:border_width, x:end] = color
            img[-border_width:, x:end] = color
            x += dash + gap
        y = 0
        while y < h:
            end = min(y + dash, h)
            img[y:end, :border_width] = color
            img[y:end, -border_width:] = color
            y += dash + gap
    return img

def apply_lens_flare(image, flare_path, flare_position=(100, 100)):
    flare = cv2.imread(flare_path, cv2.IMREAD_UNCHANGED)
    if flare is None:
        raise FileNotFoundError(f"Файл блика не найден: {flare_path}")

    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    fh, fw = flare.shape[:2]
    x0, y0 = flare_position
    x1, y1 = max(0, x0 - fw//2), max(0, y0 - fh//2)
    x2, y2 = min(w, x1 + fw), min(h, y1 + fh)

    if x2 <= x1 or y2 <= y1:
        return result.astype(np.uint8)

    flare_crop = flare[:y2-y1, :x2-x1]
    if flare_crop.shape[2] == 4:
        alpha = flare_crop[:, :, 3:4].astype(np.float32) / 255.0
        flare_rgb = flare_crop[:, :, :3].astype(np.float32)
        result[y1:y2, x1:x2] = result[y1:y2, x1:x2] * (1 - alpha) + flare_rgb * alpha
    else:
        result[y1:y2, x1:x2] = flare_crop

    return np.clip(result, 0, 255).astype(np.uint8)

def apply_watercolor_texture(image, texture_path):
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        raise FileNotFoundError(f"Текстура не найдена: {texture_path}")

    texture = resize_image(texture, image.shape[1], image.shape[0])
    result = image.astype(np.float32)

    if texture.shape[2] == 4:
        alpha = texture[:, :, 3:4].astype(np.float32) / 255.0
        tex_rgb = texture[:, :, :3].astype(np.float32)
        result = result * (1 - alpha * 0.7) + tex_rgb * (alpha * 0.7)
    else:
        gray = np.mean(texture, axis=2, keepdims=True)
        mask = gray / 255.0
        result = result * (0.9 + 0.1 * mask)

    return np.clip(result, 0, 255).astype(np.uint8)
