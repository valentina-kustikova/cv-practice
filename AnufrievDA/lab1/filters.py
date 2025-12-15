import cv2
import numpy as np
import os

def resize_image(image, width, height):
    """1. Изменяет разрешение, используя только базовые операции NumPy."""
    h0, w0, _ = image.shape
    
    if width <= 0 or height <= 0:
        raise ValueError("Высота и ширина должны быть положительными")

    x_indices = (np.arange(width) * w0 / width).astype(int)
    y_indices = (np.arange(height) * h0 / height).astype(int)

    resized_image = np.take(image, y_indices, axis=0)
    resized_image = np.take(resized_image, x_indices, axis=1)
    
    return resized_image

def apply_sepia(image, intensity=1.0):
    """2. Применяет сепию через прямое матричное умножение."""
    # Матрица для BGR формата
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    
    M = (1.0 - intensity) * np.eye(3) + intensity * sepia_matrix
    
    sepia_float = image.astype(np.float32) @ M.T
    
    return np.clip(sepia_float, 0, 255).astype(np.uint8)

def apply_vignette(image, strength=0.8):
    """3. Применяет виньетку, генерируя маску вручную."""
    h, w, _ = image.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    center_x, center_y = w / 2, h / 2
    dist = np.sqrt(((x_coords - center_x) ** 2) + ((y_coords - center_y) ** 2))
    
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist_norm = dist / max_dist
    
    mask = 1 - (dist_norm * strength)
    mask = np.clip(mask, 0, 1)
    
    vignette_image = image.astype(np.float32) * mask[..., np.newaxis]
    
    return vignette_image.astype(np.uint8)

def pixelate_area(image, x, y, w, h, pixel_size=20):
    """4. Пикселизирует область, используя базовую функцию resize."""
    h0, w0, _ = image.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w0, x + w), min(h0, y + h)
    
    if x1 <= x0 or y1 <= y0:
        return image

    output = image.copy()
    roi = output[y0:y1, x0:x1]
    
    small_w = max(1, roi.shape[1] // pixel_size)
    small_h = max(1, roi.shape[0] // pixel_size)
    small_roi = resize_image(roi, width=small_w, height=small_h)
    
    pixelated_roi = resize_image(small_roi, width=roi.shape[1], height=roi.shape[0])
    
    output[y0:y1, x0:x1] = pixelated_roi
    return output

def add_simple_frame(image, thickness, color):
    """5. Добавляет простую рамку через срезы NumPy (без изменений)."""
    output = image.copy()
    h, w = output.shape[:2]
    output[0:thickness, :] = color
    output[h-thickness:h, :] = color
    output[:, 0:thickness] = color
    output[:, w-thickness:w] = color
    return output

def add_image_frame(image, frame_path):
    """6. Накладывает рамку из PNG, используя ручное альфа-смешивание."""
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(f"Не удалось загрузить файл рамки: {frame_path}")

    frame = resize_image(frame, width=image.shape[1], height=image.shape[0])
    
    frame_rgb = frame[..., :3].astype(np.float32)
    alpha = frame[..., 3].astype(np.float32) / 255.0
    alpha_mask = alpha[..., np.newaxis]

    base = image.astype(np.float32)
    blended = frame_rgb * alpha_mask + base * (1.0 - alpha_mask)
    
    return np.clip(blended, 0, 255).astype(np.uint8)

def add_image_flare(image, flare_path, center_x, center_y, scale=1.0):
    """7. Добавляет блик с позиционированием и масштабированием."""
    flare = cv2.imread(flare_path)
    if flare is None:
        raise FileNotFoundError(f"Не удалось загрузить файл блика: {flare_path}")

    flare_h, flare_w = flare.shape[:2]
    scaled_w, scaled_h = int(flare_w * scale), int(flare_h * scale)
    if scaled_w == 0 or scaled_h == 0: return image.copy() # Слишком маленький масштаб
    flare = resize_image(flare, width=scaled_w, height=scaled_h)
    
    flare_h, flare_w = flare.shape[:2]
    
    x0 = center_x - flare_w // 2
    y0 = center_y - flare_h // 2
    
    img_h, img_w = image.shape[:2]
    overlap_x0 = max(0, x0)
    overlap_y0 = max(0, y0)
    overlap_x1 = min(img_w, x0 + flare_w)
    overlap_y1 = min(img_h, y0 + flare_h)

    if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
        return image.copy()

    image_roi = image[overlap_y0:overlap_y1, overlap_x0:overlap_x1]

    flare_x0_in_flare = overlap_x0 - x0
    flare_y0_in_flare = overlap_y0 - y0
    flare_roi = flare[flare_y0_in_flare:flare_y0_in_flare + image_roi.shape[0], 
                      flare_x0_in_flare:flare_x0_in_flare + image_roi.shape[1]]

    image_roi_float = image_roi.astype(np.float32) / 255.0
    flare_roi_float = flare_roi.astype(np.float32) / 255.0
    blended_roi_float = 1.0 - (1.0 - image_roi_float) * (1.0 - flare_roi_float)
    blended_roi = np.clip(blended_roi_float * 255, 0, 255).astype(np.uint8)

    output = image.copy()
    output[overlap_y0:overlap_y1, overlap_x0:overlap_x1] = blended_roi
    
    return output

def apply_paper_texture(image, paper_path, strength=0.5):
    """8. Накладывает текстуру бумаги через ручное смешивание по маске."""
    paper = cv2.imread(paper_path)
    if paper is None:
        raise FileNotFoundError(f"Не удалось загрузить файл текстуры: {paper_path}")

    paper = resize_image(paper, width=image.shape[1], height=image.shape[0])

    paper_gray = paper.astype(np.float32).mean(axis=2) / 255.0
    mask = (1.0 - paper_gray) * strength
    mask = mask[..., np.newaxis]

    img_f = image.astype(np.float32)
    tex_f = paper.astype(np.float32)

    blended = img_f * (1.0 - mask) + tex_f * mask
    return np.clip(blended, 0, 255).astype(np.uint8)