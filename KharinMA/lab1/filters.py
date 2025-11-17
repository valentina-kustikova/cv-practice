import cv2
import numpy as np


def resize_image(image, width=None, height=None, scale=None):
    h, w = image.shape[:2]
    
    if scale is not None:
        # Масштабирование по коэффициенту
        new_width = int(w * scale)
        new_height = int(h * scale)
    elif width is not None and height is not None:
        # Задана новая ширина и высота
        new_width = width
        new_height = height
    elif width is not None:
        # Задана только ширина, сохраняем пропорции
        new_width = width
        new_height = int(h * (width / w))
    elif height is not None:
        # Задана только высота, сохраняем пропорции
        new_height = height
        new_width = int(w * (height / h))
    else:
        return image
    
    # Создаем новое изображение
    resized = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    # Вычисляем коэффициенты масштабирования
    x_ratio = w / new_width
    y_ratio = h / new_height
    
    # Билинейная интерполяция
    for i in range(new_height):
        for j in range(new_width):
            # Вычисляем соответствующие координаты в исходном изображении
            x = j * x_ratio
            y = i * y_ratio
            
            # Находим соседние пиксели
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, w - 1)
            y2 = min(y1 + 1, h - 1)
            
            # Вычисляем веса для интерполяции
            dx = x - x1
            dy = y - y1
            
            # Билинейная интерполяция для каждого канала
            for c in range(image.shape[2]):
                value = (image[y1, x1, c] * (1 - dx) * (1 - dy) +
                        image[y1, x2, c] * dx * (1 - dy) +
                        image[y2, x1, c] * (1 - dx) * dy +
                        image[y2, x2, c] * dx * dy)
                resized[i, j, c] = int(value)
    
    return resized


def apply_sepia(image, intensity=1.0):
    # Создаем копию изображения
    sepia_image = np.copy(image).astype(np.float32)
    
    # Матрица преобразования для сепии (BGR формат)
    # Применяем преобразование для каждого пикселя
    for i in range(sepia_image.shape[0]):
        for j in range(sepia_image.shape[1]):
            b, g, r = sepia_image[i, j]
            
            # Формулы для эффекта сепии
            new_r = min(255, 0.393 * r + 0.769 * g + 0.189 * b)
            new_g = min(255, 0.349 * r + 0.686 * g + 0.168 * b)
            new_b = min(255, 0.272 * r + 0.534 * g + 0.131 * b)
            
            # Применяем интенсивность
            sepia_image[i, j, 0] = b * (1 - intensity) + new_b * intensity
            sepia_image[i, j, 1] = g * (1 - intensity) + new_g * intensity
            sepia_image[i, j, 2] = r * (1 - intensity) + new_r * intensity
    
    return sepia_image.astype(np.uint8)


def apply_vignette(image, strength=0.5):
    h, w = image.shape[:2]
    
    # Создаем маску виньетки
    center_x, center_y = w // 2, h // 2
    
    # Максимальное расстояние от центра
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # Создаем маску
    mask = np.zeros((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            # Вычисляем расстояние от центра
            distance = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            
            # Нормализуем расстояние и применяем затемнение
            normalized_distance = distance / max_distance
            mask[i, j] = 1 - (normalized_distance * strength) # если normalized_distance == 0 оставляем пиксель без изменений
    
    # Применяем маску к каждому каналу
    vignette_image = np.copy(image).astype(np.float32)
    for c in range(image.shape[2]):
        vignette_image[:, :, c] = vignette_image[:, :, c] * mask
    
    return vignette_image.astype(np.uint8)


def pixelate_region(image, x, y, width, height, pixel_size=10):
    result = np.copy(image)
    
    # Ограничиваем область границами изображения
    x_end = min(x + width, image.shape[1])
    y_end = min(y + height, image.shape[0])
    
    # Проходим по области блоками pixel_size x pixel_size
    for i in range(y, y_end, pixel_size):
        for j in range(x, x_end, pixel_size):
            # Определяем границы текущего блока
            block_y_end = min(i + pixel_size, y_end)
            block_x_end = min(j + pixel_size, x_end)
            
            # Вычисляем средний цвет блока
            block = image[i:block_y_end, j:block_x_end]
            avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            
            # Заполняем блок средним цветом
            result[i:block_y_end, j:block_x_end] = avg_color
    
    return result


def add_simple_frame(image, frame_width=20, color=(0, 0, 0)):
    result = np.copy(image)
    h, w = image.shape[:2]
    
    # Верхняя и нижняя рамка
    result[0:frame_width, :] = color
    result[h-frame_width:h, :] = color
    
    # Левая и правая рамка
    result[:, 0:frame_width] = color
    result[:, w-frame_width:w] = color
    
    return result


def add_decorative_frame(image, frame_width=30, color=(255, 215, 0), pattern='waves'):
    result = np.copy(image)
    h, w = image.shape[:2]
    
    if pattern == 'waves':
        for i in range(h):
            for j in range(w):
                # Вычисляем волновые смещения
                wave_offset_horizontal = int(5 * np.sin(j * 0.1))
                wave_offset_vertical = int(5 * np.sin(i * 0.1))
                
                # Проверяем, попадает ли пиксель в ЛЮБУЮ из четырех границ
                is_top = i < frame_width + wave_offset_horizontal
                is_bottom = i > h - frame_width - wave_offset_horizontal
                is_left = j < frame_width + wave_offset_vertical
                is_right = j > w - frame_width - wave_offset_vertical
                
                # Красим пиксель только один раз, если он в любой из зон
                if is_top or is_bottom or is_left or is_right:
                    result[i, j] = color
    
    elif pattern == 'zigzag':
        # Зигзагообразный узор
        for i in range(h):
            for j in range(w):
                # Вычисляем смещения для горизонтальных и вертикальных границ
                zigzag_offset_horizontal = int(abs((j % 40) - 20))
                zigzag_offset_vertical = int(abs((i % 40) - 20))
                
                # Проверяем все четыре границы
                is_top = i < frame_width + zigzag_offset_horizontal
                is_bottom = i > h - frame_width - zigzag_offset_horizontal
                is_left = j < frame_width + zigzag_offset_vertical
                is_right = j > w - frame_width - zigzag_offset_vertical
                
                # Красим пиксель только один раз, если он в любой из зон
                if is_top or is_bottom or is_left or is_right:
                    result[i, j] = color
    
    return result


def add_lens_flare(image, center_x=None, center_y=None, intensity=0.8):
    import os
    
    h, w = image.shape[:2]
    
    # Загружаем текстуру блика с альфа-каналом
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blik_path = os.path.join(script_dir, 'blik.png')
    
    if not os.path.exists(blik_path):
        print(f"Предупреждение: текстура блика не найдена: {blik_path}")
        return image
    
    blik = cv2.imread(blik_path, cv2.IMREAD_UNCHANGED)
    if blik is None:
        print(f"Предупреждение: не удалось загрузить текстуру блика: {blik_path}")
        return image
    
    # Подгоняем размер текстуры под размер изображения
    blik = cv2.resize(blik, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Проверяем наличие альфа-канала
    if blik.shape[2] == 4:
        # Извлекаем альфа-канал и нормализуем
        alpha = blik[:, :, 3] / 255.0
        blik_rgb = blik[:, :, :3].astype(np.float32)
        img_f = image.astype(np.float32)
        
        # Смешиваем изображения с учетом альфа-канала
        for c in range(3):
            img_f[:, :, c] = img_f[:, :, c] * (1 - alpha * intensity) + blik_rgb[:, :, c] * (alpha * intensity)
        
        result = np.clip(img_f, 0, 255).astype(np.uint8)
    else:
        # Если альфа-канала нет, используем простое смешивание
        blik_rgb = blik.astype(np.float32)
        img_f = image.astype(np.float32)
        result = img_f * (1 - intensity) + blik_rgb * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def add_watercolor_texture(image, intensity=0.3):
    import os
    
    # Загружаем текстуру акварельной бумаги
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paper_path = os.path.join(script_dir, 'paper.png')
    
    if not os.path.exists(paper_path):
        print(f"Предупреждение: текстура бумаги не найдена: {paper_path}")
        return image
    
    paper_texture = cv2.imread(paper_path)
    if paper_texture is None:
        print(f"Предупреждение: не удалось загрузить текстуру бумаги: {paper_path}")
        return image
    
    h, w = image.shape[:2]
    
    # Подгоняем размер текстуры под размер изображения
    paper_resized = cv2.resize(paper_texture, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Конвертируем изображения в float32 для точных вычислений
    result = image.astype(np.float32)
    paper_float = paper_resized.astype(np.float32)
    
    # Складываем изображение с текстурой с учетом интенсивности
    result = result + paper_float * intensity
    
    # Обрезаем значения до диапазона [0, 255]
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


# Словарь доступных фильтров для удобного использования
FILTERS = {
    'resize': resize_image,
    'sepia': apply_sepia,
    'vignette': apply_vignette,
    'pixelate': pixelate_region,
    'simple_frame': add_simple_frame,
    'decorative_frame': add_decorative_frame,
    'lens_flare': add_lens_flare,
    'watercolor': add_watercolor_texture
}
