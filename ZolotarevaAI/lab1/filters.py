import cv2
import numpy as np


def resize_image(image, scale = 0.5):
    """Изменение разрешения изображения"""
    #return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # Интерполяция по ближайшему соседу
    # h=100, new_h=50, то y_idx = [0, 2, 4, 6, ..., 98]
    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)

    resized = image[y_idx][:, x_idx]

    return resized


def apply_sepia(image):
    """Фотоэффект сепии (матричное преобразование цветов)"""
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = image @ sepia_matrix.T
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def apply_vignette(image, strength=0.5):
    """Фотоэффект виньетки (затемнение краёв вручную через радиальную маску)"""
    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)

    # создаём радиальную маску от 1 (в центре) до strength (по краям)
    mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask[y, x] = 1 - strength * (dist / max_dist)
    mask = np.clip(mask, 0, 1)

    vignette = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        vignette[:, :, c] = (image[:, :, c] * mask).astype(np.uint8)
    return vignette


def pixelate_region(image, x, y, w, h, pixel_size=10):
    """Пикселизация заданной прямоугольной области"""
    result = image.copy()
    roi = result[y:y+h, x:x+w]

    small = cv2.resize(roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    result[y:y+h, x:x+w] = pixelated

    return result


def add_rect_frame(image, color=(0, 0, 255), thickness=20):
    """Наложение одноцветной прямоугольной рамки"""
    framed = image.copy()
    h, w = image.shape[:2]
    # Верхняя и нижняя рамки
    framed[0:thickness, :, :] = color
    framed[h-thickness:h, :, :] = color
    # Левая и правая рамки
    framed[:, 0:thickness, :] = color
    framed[:, w-thickness:w, :] = color
    return framed



def add_shape_frame(image, texture_path, thickness=50, intensity=1.0): #intensity - коэф смешивания
    """Текстурная рамка"""
    h, w = image.shape[:2]
    # Загружаем текстуру рамки
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        print("Ошибка: текстура не найдена!")
        return image

    # Масштабируем текстуру под размеры изображения
    texture = cv2.resize(texture, (w, h))
    
    # Создаём маску рамки: 1 — рамка, 0 — центр
    #Срезы и массивы в питоне: [строки,столбцы] :a - от начала до а-1, a: - от а до конца
    mask = np.zeros((h, w), dtype=np.float32) #массив нулей
    mask[:thickness, :] = 1           # верх
    mask[h-thickness:, :] = 1         # низ
    mask[:, :thickness] = 1           # левая
    mask[:, w-thickness:] = 1         # правая
    mask = mask[:, :, None]  # делаем 3 канала для RGB

    texture_rgb = texture.astype(np.float32)
    alpha = mask * intensity
    result = image.astype(np.float32) * (1 - alpha) + texture_rgb * alpha

    return np.clip(result, 0, 255).astype(np.uint8)

def add_lens_flare(image, flare_texture_path, intensity=0.5):
    """Наложение бликов через текстуру с альфа-каналом"""
    flare = cv2.imread(flare_texture_path, cv2.IMREAD_UNCHANGED)
    flare = cv2.resize(flare, (image.shape[1], image.shape[0]))
    # Разделяем каналы
    if flare.shape[2] == 4:
        b, g, r, a = cv2.split(flare)
        alpha = (a.astype(np.float32) / 255.0 * intensity)[:, :, None]
        flare_rgb = cv2.merge([b, g, r])
        result = (image.astype(np.float32) * (1 - alpha) + flare_rgb.astype(np.float32) * alpha)
    else:
        result = cv2.addWeighted(image, 1 - intensity, flare, intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_paper_texture(image, texture_path, intensity=0.3):
    """Наложение текстуры акварельной бумаги с альфа-каналом"""
    texture = cv2.imread(texture_path)
    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    texture = texture.astype(np.float32)
    image = image.astype(np.float32)

    result = cv2.addWeighted(image, 1 - intensity, texture, intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)
