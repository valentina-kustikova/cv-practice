# watercolor_paper.py
import numpy as np
from PIL import Image


def apply_watercolor_paper(image):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        image = image.astype(np.float32)

    h, w = image.shape[:2]
    texture_path = "textures/watercolor_texture.jpg"
    texture_img = Image.open(texture_path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    texture = np.asarray(texture_img, dtype=np.float32)

    #  Нормализуем текстуру
    gray = np.mean(texture, axis=2, keepdims=True)
    gray_norm = gray / np.mean(gray)
    texture_effect = np.clip(gray_norm, 0.8, 1.2)

    result = image * texture_effect

    return np.clip(result, 0, 255).astype(np.uint8)
