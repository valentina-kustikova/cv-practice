import numpy as np
import cv2
import os

TEXTURES_DIR = "textures"

def _load_texture(filename):
    path = os.path.join(TEXTURES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Текстура не найдена: {path}")
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

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
    
    # Матричный подход: создаем сетку координат и используем индексацию
    y_coords = np.arange(height, dtype=np.float32) * (h / height)
    x_coords = np.arange(width, dtype=np.float32) * (w / width)
    
    # Ограничиваем координаты границами изображения
    y_coords = np.clip(y_coords, 0, h - 1).astype(np.int32)
    x_coords = np.clip(x_coords, 0, w - 1).astype(np.int32)
    
    # Создаем матрицы индексов для всех пикселей одновременно
    y_indices, x_indices = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Матричная индексация для получения всех пикселей сразу
    resized = image[y_indices, x_indices]
    
    return resized

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    
    # Матричный подход: преобразуем изображение в формат (height*width, 3)
    # и применяем матричное умножение ко всем пикселям одновременно
    h, w = image.shape[:2]
    image_float = image.astype(np.float32)
    
    # Преобразуем изображение в матрицу (h*w, 3)
    pixels = image_float.reshape(-1, 3)
    
    # Применяем матричное умножение: (h*w, 3) @ (3, 3) = (h*w, 3)
    sepia_pixels = pixels @ sepia_filter.T
    
    # Ограничиваем значения и преобразуем обратно в форму изображения
    sepia_pixels = np.clip(sepia_pixels, 0, 255)
    result = sepia_pixels.reshape(h, w, 3).astype(np.uint8)
    
    return result

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

def apply_lens_flare(image, flare_position=(100, 100)):
    flare = None
    # Список путей для поиска файла блика
    search_paths = [TEXTURES_DIR, "."]  # Сначала в textures, потом в корневой папке
    # Приоритет: сначала PNG (может иметь альфа-канал), потом JPG
    extensions = [".png", ".jpg", ".jpeg"]
    
    for search_path in search_paths:
        for ext in extensions:
            filename = "glare" + ext
            path = os.path.join(search_path, filename)
            if os.path.exists(path):
                flare = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if flare is not None:
                    break
        if flare is not None:
            break
    
    if flare is None:
        raise FileNotFoundError(f"Файл блика не найден (glare.png или glare.jpg в папке {TEXTURES_DIR} или в корневой папке)")

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
        # Обработка изображения с альфа-каналом (PNG с прозрачностью)
        alpha = flare_crop[:, :, 3:4].astype(np.float32) / 255.0
        flare_rgb = flare_crop[:, :, :3].astype(np.float32)
        result[y1:y2, x1:x2] = result[y1:y2, x1:x2] * (1 - alpha) + flare_rgb * alpha
    else:
        # Обработка изображения без альфа-канала (JPG)
        # Используем screen blend mode для бликов (более естественное наложение)
        flare_rgb = flare_crop.astype(np.float32)
        base = result[y1:y2, x1:x2].astype(np.float32)
        # Screen blend: 1 - (1 - base) * (1 - overlay)
        blended = 255.0 - (255.0 - base) * (255.0 - flare_rgb) / 255.0
        # Смешиваем с исходным изображением для более мягкого эффекта
        blend_factor = 0.8
        result[y1:y2, x1:x2] = np.clip(
            base * (1 - blend_factor) + blended * blend_factor,
            0, 255
        ).astype(np.uint8)

    return np.clip(result, 0, 255).astype(np.uint8)

def apply_watercolor_texture(image, strength=0.8):
    texture = None
    for ext in [".png", ".jpg", ".jpeg"]:
        try:
            texture = _load_texture("watercolor_paper" + ext)
            break
        except FileNotFoundError:
            continue
    if texture is None:
        raise FileNotFoundError(f"Файл акварели не найден в папке {TEXTURES_DIR} (watercolor_paper.png или .jpg)")

    texture = resize_image(texture, image.shape[1], image.shape[0])
    result = image.astype(np.float32)
    tex_float = texture.astype(np.float32)

    if texture.shape[2] == 4:
        alpha = tex_float[:, :, 3:4] / 255.0
        tex_rgb = tex_float[:, :, :3]
        blend = alpha * strength * 1.2
        blend = np.clip(blend, 0, 1)
        result = result * (1 - blend) + tex_rgb * blend
    else:
        tex_rgb = tex_float
        gray = np.mean(tex_rgb, axis=2, keepdims=True)
        mask = gray / 255.0
        
        for c in range(3):
            channel = result[:, :, c:c+1]
            tex_channel = tex_rgb[:, :, c:c+1]
            overlay = np.where(channel < 128,
                              (2 * channel * tex_channel) / 255.0,
                              255.0 - 2 * (255.0 - channel) * (255.0 - tex_channel) / 255.0)
            enhanced_strength = strength * 1.1
            enhanced_strength = min(enhanced_strength, 1.0)
            result[:, :, c:c+1] = channel * (1 - enhanced_strength) + overlay * enhanced_strength

    return np.clip(result, 0, 255).astype(np.uint8)