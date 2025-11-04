#lens_flare
import numpy as np
from PIL import Image
import os


def apply_lens_flare(image):
    h, w = image.shape[:2]

    texture_path = os.path.join(os.path.dirname(__file__), '..', 'textures', 'glare.jpg')
    try:
        texture_pil = Image.open(texture_path).convert("RGB")
        texture = np.asarray(texture_pil, dtype=np.uint8)
        texture = resize_texture(texture, w, h, scale=1.9)
    except Exception as e:
        raise ValueError(f"Не удалось загрузить текстуру: {str(e)}")

    # позиция блика
    strength = 0.5
    center_x, center_y = 3 * (w // 4), h // 4
    h_glare, w_glare = texture.shape[:2]

    y_start = center_y - h_glare // 2
    y_end = y_start + h_glare
    x_start = center_x - w_glare // 2
    x_end = x_start + w_glare

    # обрезка
    img_y_start = max(0, y_start)
    img_y_end = min(h, y_end)
    img_x_start = max(0, x_start)
    img_x_end = min(w, x_end)

    glare_y_start = max(0, -y_start)
    glare_y_end = h_glare - max(0, y_end - h)
    glare_x_start = max(0, -x_start)
    glare_x_end = w_glare - max(0, x_end - w)

    result = image.astype(np.float32)
    glare_region = texture[glare_y_start:glare_y_end, glare_x_start:glare_x_end].astype(np.float32) * strength
    result[img_y_start:img_y_end, img_x_start:img_x_end] += glare_region

    return np.clip(result, 0, 255).astype(np.uint8)


def resize_texture(texture, target_w, target_h, scale):
    h, w = texture.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)

    resized = texture[y_idx[:, None], x_idx[None, :]]

    return resized
