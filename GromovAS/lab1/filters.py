import cv2
import numpy as np
import sys

# Фильтры
# Изменение размера
def resize_image(image, scale_x=1.0, scale_y=1.0):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_y), int(w * scale_x)
    resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            orig_i = min(int(i / scale_y), h - 1)
            orig_j = min(int(j / scale_x), w - 1)
            resized[i, j] = image[orig_i, orig_j]
    return resized

# Сепия
def sepia_effect(image):
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    img_float = image.astype(np.float32)
    sepia_img = np.dot(img_float.reshape(-1, 3), sepia_matrix.T).reshape(img_float.shape)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

# Виньетка
def vignette_effect(image, strength=0.5):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = np.exp(-(((X - w/2)**2)/(2*(w*strength)**2) + ((Y - h/2)**2)/(2*(h*strength)**2)))
    mask = mask / mask.max()
    vignette = image.astype(np.float32)
    for c in range(3):
        vignette[:,:,c] *= mask
    return np.clip(vignette, 0, 255).astype(np.uint8)

# Пикселизация
def pixelate_region(image, x, y, w, h, pixel_size=10):
    img_copy = image.copy()
    x_end = min(x + w, image.shape[1])
    y_end = min(y + h, image.shape[0])
    for i in range(y, y_end, pixel_size):
        for j in range(x, x_end, pixel_size):
            block = img_copy[i:i+pixel_size, j:j+pixel_size]
            avg_color = np.mean(block, axis=(0,1)).astype(np.uint8)
            img_copy[i:i+pixel_size, j:j+pixel_size] = avg_color
    return img_copy

# Прямоугольная граница
def add_rectangular_border(image, color=(0,255,0), thickness=30):
    bordered = image.copy()
    bordered[:thickness, :] = color
    bordered[-thickness:, :] = color
    bordered[:, :thickness] = color
    bordered[:, -thickness:] = color
    return bordered

# Фигурная граница
def add_shape_border(image, color=(255,0,0), thickness=25, shape_type='ellipse'):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h//2, w//2
    if shape_type == 'ellipse':
        for i in range(h):
            for j in range(w):
                y_norm = (i - cy)/(h/2 - thickness)
                x_norm = (j - cx)/(w/2 - thickness)
                if x_norm**2 + y_norm**2 >= 1:
                    mask[i,j] = 1
    elif shape_type == 'circle':
        radius = min(h,w)//2 - thickness
        for i in range(h):
            for j in range(w):
                if (i-cy)**2 + (j-cx)**2 >= radius**2:
                    mask[i,j] = 1
    elif shape_type == 'diamond':
        for i in range(h):
            for j in range(w):
                if abs(i-cy) + abs(j-cx) >= min(h,w)//2 - thickness:
                    mask[i,j] = 1
    bordered = image.copy()
    for c in range(3):
        bordered[:,:,c] = np.where(mask==1, color[c], bordered[:,:,c])
    return bordered

# Линза
def lens_flare_effect(img, flare_center=(0.5,0.5), intensity=1.0):
    h, w = img.shape[:2]
    cx, cy = int(flare_center[0]*w), int(flare_center[1]*h)
    Y, X = np.ogrid[:h, :w]
    mask = np.exp(-(np.sqrt((X - cx)**2 + (Y - cy)**2) / (0.25 * max(h,w)))**2)
    mask = mask[:,:,np.newaxis]
    flare_color = np.array([1.0,0.8,0.6])
    flare_img = img.astype(np.float32)/255.0
    flare_img += mask * flare_color * intensity
    return np.clip(flare_img*255, 0, 255).astype(np.uint8)

# Бумага
def watercolor_texture(image, scale=12, intensity=0.4):
    h, w = image.shape[:2]
    base_noise = np.random.rand(h // scale + 2, w // scale + 2) * 0.8 + 0.2
    noise = cv2.resize(base_noise, (w, h), interpolation=cv2.INTER_LINEAR)
    kernel_size = max(3, scale // 2 * 2 + 1)
    smoothed = np.zeros_like(noise)
    offset = kernel_size // 2
    padded = np.pad(noise, offset, mode='edge')
    for y in range(h):
        for x in range(w):
            smoothed[y, x] = np.mean(padded[y:y+kernel_size, x:x+kernel_size])
    texture = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    texture = np.repeat(texture[:, :, np.newaxis], 3, axis=2)
    img_float = image.astype(np.float32) / 255.0
    img_textured = img_float * (1 - intensity * 0.7 + texture * intensity * 1.3)
    return np.clip(img_textured * 255, 0, 255).astype(np.uint8)

# Функция вызова фильтра
def apply_filter(image, filter_type, *args):
    if filter_type == 'resize':
        scale_x = float(args[0]) if len(args) > 0 else 1.0
        scale_y = float(args[1]) if len(args) > 1 else scale_x
        return resize_image(image, scale_x, scale_y)
    elif filter_type == 'sepia':
        return sepia_effect(image)
    elif filter_type == 'vignette':
        strength = float(args[0]) if len(args) > 0 else 0.5
        return vignette_effect(image, strength)
    elif filter_type == 'pixelate':
        x, y, w, h = map(int, args[:4])
        pixel_size = int(args[4]) if len(args) > 4 else 10
        return pixelate_region(image, x, y, w, h, pixel_size)
    elif filter_type == 'rect_border':
        thickness = int(args[0]) if len(args) > 0 else 20
        color = tuple(map(int, args[1:4])) if len(args) >= 4 else (0,255,0)
        return add_rectangular_border(image, color, thickness)
    elif filter_type == 'shape_border':
        thickness = int(args[0]) if len(args) > 0 else 20
        color = tuple(map(int, args[1:4])) if len(args) >= 4 else (255,0,0)
        shape_type = args[4] if len(args) > 4 else 'ellipse'
        return add_shape_border(image, color, thickness, shape_type)
    elif filter_type == 'lens_flare':
        flare_center = tuple(map(float, args[:2])) if len(args) >= 2 else (0.5,0.5)
        intensity = float(args[2]) if len(args) > 2 else 1.0
        return lens_flare_effect(image, flare_center, intensity)
    elif filter_type == 'watercolor':
        scale = int(args[0]) if len(args) > 0 else 12
        intensity = float(args[1]) if len(args) > 1 else 0.4
        return watercolor_texture(image, scale, intensity)
    else:
        raise ValueError("Неизвестный фильтр")

# Main
def main():
    if len(sys.argv) < 3:
        print("Использование: python filters.py <image_path> <filter_type> [filter_params...]")
        print("Пример: python filters.py image.jpg sepia")
        print("Пример: python filters.py image.jpg pixelate 100 100 200 200 15")
        return

    image_path = sys.argv[1]
    filter_type = sys.argv[2]
    filter_params = sys.argv[3:]

    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: изображение не найдено")
        return

    try:
        filtered = apply_filter(image, filter_type, *filter_params)
    except Exception as e:
        print("Ошибка при применении фильтра:", e)
        return

    cv2.imshow("Original", image)
    cv2.imshow(f"Filtered: {filter_type}", filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
