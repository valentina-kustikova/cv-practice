import cv2
import numpy as np
import argparse
import sys

def argument_parser():
    """Парсер аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Image processing filters')
    parser.add_argument('-i', '--image', help='Path to input image', type=str, required=True)
    parser.add_argument('-o', '--output', help='Path to output image', type=str, default='output.jpg')
    parser.add_argument('-f', '--filter', help='Filter to apply',
                        choices=['resize', 'sepia', 'vignette', 'pixelate', 'border', 'fancy_border', 'lens_flare', 'watercolor'],
                        required=True)
    
    # Параметры для различных фильтров
    parser.add_argument('--scale', type=float, help='Scale factor for resize')
    parser.add_argument('--width', type=int, help='New width for resize')
    parser.add_argument('--height', type=int, help='New height for resize')
    
    parser.add_argument('--sepia_intensity', type=float, default=1.0, help='Intensity for sepia filter')
    
    parser.add_argument('--vignette_strength', type=float, default=0.5, help='Strength for vignette')
    parser.add_argument('--vignette_radius', type=float, default=0.8, help='Radius for vignette')
    
    parser.add_argument('--pixel_size', type=int, default=10, help='Pixel size for pixelation')
    parser.add_argument('--pixel_x', type=int, help='X coordinate for pixelation area')
    parser.add_argument('--pixel_y', type=int, help='Y coordinate for pixelation area')
    parser.add_argument('--pixel_width', type=int, help='Width for pixelation area')
    parser.add_argument('--pixel_height', type=int, help='Height for pixelation area')
    
    parser.add_argument('--border_width', type=int, default=20, help='Border width')
    parser.add_argument('--border_color', type=str, default='0,0,0', help='Border color as R,G,B')
    
    parser.add_argument('--fancy_border_type', choices=['wave', 'zigzag', 'triangle'], help='Type of fancy border')
    parser.add_argument('--fancy_border_color', type=str, default='255,0,0', help='Fancy border color as R,G,B')
    
    parser.add_argument('--flare_x', type=float, default=0.8, help='X coordinate for lens flare (0-1)')
    parser.add_argument('--flare_y', type=float, default=0.2, help='Y coordinate for lens flare (0-1)')
    parser.add_argument('--flare_intensity', type=float, default=1.0, help='Intensity for lens flare')
    
    parser.add_argument('--watercolor_intensity', type=float, default=0.3, help='Intensity for watercolor effect')
    
    return parser.parse_args()

def load_image(image_path):
    """Загрузка изображения"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    return image

def show_images(original, result, filter_name):
    """Отображение исходного и обработанного изображения"""
    cv2.imshow('Original Image', original)
    cv2.imshow(f'Result - {filter_name}', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, output_path):
    """Сохранение изображения"""
    cv2.imwrite(output_path, image)

# Реализации фильтров

def resize_image(image, scale=None, new_width=None, new_height=None):
    """Изменение размера изображения"""
    h, w = image.shape[:2]
    
    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif new_width is not None and new_height is not None:
        new_w = new_width
        new_h = new_height
    elif new_width is not None:
        new_h = int(h * new_width / w)
        new_w = new_width
    elif new_height is not None:
        new_w = int(w * new_height / h)
        new_h = new_height
    else:
        return image.copy()
    
    # Метод ближайшего соседа
    scale_x = w / new_w
    scale_y = h / new_h
    
    y_indices = (np.arange(new_h) * scale_y).astype(np.int32)
    x_indices = (np.arange(new_w) * scale_x).astype(np.int32)
    
    y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
    result = image[y_grid, x_grid]
    
    return result

def sepia_filter(image, intensity=1.0):
    """Фильтр сепии"""
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    
    # Применяем матричное преобразование
    result = image.astype(np.float32) @ sepia_matrix.T
    result = np.clip(result, 0, 255)
    
    # Смешиваем с оригиналом
    if intensity < 1.0:
        result = image * (1 - intensity) + result * intensity
        result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)

def vignette_filter(image, strength=0.5, radius=0.8):
    """Эффект виньетки"""
    h, w = image.shape[:2]
    
    # Создаем координатные сетки
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2
    
    # Вычисляем расстояние от центра
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Нормализуем расстояние
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distance = distance / (max_distance * radius)
    
    # Создаем маску виньетки
    vignette_mask = 1 - normalized_distance
    vignette_mask = np.clip(vignette_mask, 0, 1)
    vignette_mask = vignette_mask ** (1 / strength)
    
    # Применяем маску
    result = image.astype(np.float32) * vignette_mask[:, :, np.newaxis]
    
    return np.clip(result, 0, 255).astype(np.uint8)

def pixelate_region(image, x, y, width, height, pixel_size):
    """Пикселизация области"""
    result = image.copy()
    
    # Ограничиваем координаты размерами изображения
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)
    
    # Обрабатываем область
    for i in range(y1, y2, pixel_size):
        for j in range(x1, x2, pixel_size):
            # Определяем границы блока
            i_end = min(i + pixel_size, y2)
            j_end = min(j + pixel_size, x2)
            
            # Извлекаем блок и вычисляем средний цвет
            block = result[i:i_end, j:j_end]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                result[i:i_end, j:j_end] = avg_color
    
    return result

def interactive_pixelation(image, pixel_size=10):
    """Интерактивная пикселизация"""
    temp_image = image.copy()
    selected_region = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_region, temp_image
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_region = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if selected_region:
                selected_region[2] = x
                selected_region[3] = y
                temp_image = image.copy()
                cv2.rectangle(temp_image, 
                            (selected_region[0], selected_region[1]),
                            (selected_region[2], selected_region[3]), 
                            (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            if selected_region:
                x1, y1, x2, y2 = selected_region
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                if width > 0 and height > 0:
                    result = pixelate_region(image, x, y, width, height, pixel_size)
                    return result
        return None
    
    cv2.imshow('Select region for pixelation (click and drag)', temp_image)
    cv2.setMouseCallback('Select region for pixelation (click and drag)', mouse_callback)
    
    print("Instructions:")
    print("- Click and drag to select region")
    print("- Press 'q' to quit without changes")
    print("- Press 'a' to apply pixelation")
    
    while True:
        cv2.imshow('Select region for pixelation (click and drag)', temp_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return image
        elif key == ord('a') and selected_region:
            x1, y1, x2, y2 = selected_region
            x = min(x1, x2)
            y = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            if width > 0 and height > 0:
                result = pixelate_region(image, x, y, width, height, pixel_size)
                cv2.destroyAllWindows()
                return result

def add_border(image, border_width, border_color):
    """Добавление прямоугольной рамки"""
    result = image.copy()
    h, w = result.shape[:2]
    
    # Закрашиваем рамку
    result[0:border_width, :] = border_color
    result[h-border_width:h, :] = border_color
    result[:, 0:border_width] = border_color
    result[:, w-border_width:w] = border_color
    
    return result

def add_fancy_border(image, border_type, border_color):
    """Добавление фигурной одноцветной рамки по краям изображения"""
    h, w = image.shape[:2]
    border_width = min(h, w) // 15 
    
    result = image.copy()
    
    if border_type == 'wave':
        # Волнистая рамка с синусоидальным паттерном
        for i in range(border_width):
            # Верхняя граница
            wave_top = border_width * (0.5 + 0.5 * np.sin(np.arange(w) * 2 * np.pi / 80))
            mask_top = i < wave_top
            result[i, mask_top] = border_color
            
            # Нижняя граница
            wave_bottom = border_width * (0.5 + 0.5 * np.sin(np.arange(w) * 2 * np.pi / 80 + np.pi))
            mask_bottom = i < wave_bottom
            result[h - 1 - i, mask_bottom] = border_color
        
        for j in range(border_width):
            # Левая граница
            wave_left = border_width * (0.5 + 0.5 * np.sin(np.arange(h) * 2 * np.pi / 80))
            mask_left = j < wave_left
            result[mask_left, j] = border_color
            
            # Правая граница
            wave_right = border_width * (0.5 + 0.5 * np.sin(np.arange(h) * 2 * np.pi / 80 + np.pi))
            mask_right = j < wave_right
            result[mask_right, w - 1 - j] = border_color
            
    elif border_type == 'zigzag':
        # Зигзагообразная рамка с пилообразным паттерном
        period = 40 
        
        for i in range(border_width):
            # Верхняя и нижняя границы
            for x in range(w):
                # Верхняя граница
                zigzag_top = border_width * (0.5 + 0.5 * abs((x % period) - period/2) / (period/2))
                if i < zigzag_top:
                    result[i, x] = border_color
                
                # Нижняя граница
                zigzag_bottom = border_width * (0.5 + 0.5 * abs((x % period) - period/2) / (period/2))
                if i < zigzag_bottom:
                    result[h - 1 - i, x] = border_color
        
        for j in range(border_width):
            # Левая и правая границы
            for y in range(h):
                # Левая граница
                zigzag_left = border_width * (0.5 + 0.5 * abs((y % period) - period/2) / (period/2))
                if j < zigzag_left:
                    result[y, j] = border_color
                
                # Правая граница
                zigzag_right = border_width * (0.5 + 0.5 * abs((y % period) - period/2) / (period/2))
                if j < zigzag_right:
                    result[y, w - 1 - j] = border_color
                    
    elif border_type == 'triangle':
        # Треугольная рамка с линейным градиентом
        for i in range(border_width):
            # Верхняя граница - треугольный паттерн
            triangle_top = border_width * (1 - np.abs(np.arange(w) - w/2) / (w/2))
            mask_top = i < triangle_top
            result[i, mask_top] = border_color
            
            # Нижняя граница - треугольный паттерн
            triangle_bottom = border_width * (1 - np.abs(np.arange(w) - w/2) / (w/2))
            mask_bottom = i < triangle_bottom
            result[h - 1 - i, mask_bottom] = border_color
        
        for j in range(border_width):
            # Левая граница - треугольный паттерн
            triangle_left = border_width * (1 - np.abs(np.arange(h) - h/2) / (h/2))
            mask_left = j < triangle_left
            result[mask_left, j] = border_color
            
            # Правая граница - треугольный паттерн
            triangle_right = border_width * (1 - np.abs(np.arange(h) - h/2) / (h/2))
            mask_right = j < triangle_right
            result[mask_right, w - 1 - j] = border_color
    
    return result

def lens_flare_effect(image, flare_x, flare_y, intensity=1.0):
    """Эффект бликов"""
    # Загружаем текстуру
    flare_texture = cv2.imread("glare.jpg")
        
    h, w = image.shape[:2]
        
    # Преобразуем координаты
    abs_x = int(flare_x * w)
    abs_y = int(flare_y * h)
        
    # Масштабируем текстуру
    flare_size = min(h, w) // 2
    flare_resized = resize_image(flare_texture, new_width=flare_size, new_height=flare_size)
        
    result = image.astype(np.float32)
    flare_img = flare_resized.astype(np.float32)
        
    # Позиционируем блик
    flare_h, flare_w = flare_img.shape[:2]
    y_start = max(0, abs_y - flare_h // 2)
    y_end = min(h, abs_y + flare_h // 2)
    x_start = max(0, abs_x - flare_w // 2)
    x_end = min(w, abs_x + flare_w // 2)
        
    # Обрезаем блик
    flare_y_start = max(0, - (abs_y - flare_h // 2))
    flare_y_end = flare_h - max(0, (abs_y + flare_h // 2) - h)
    flare_x_start = max(0, - (abs_x - flare_w // 2))
    flare_x_end = flare_w - max(0, (abs_x + flare_w // 2) - w)
        
    # Накладываем блик
    flare_region = flare_img[flare_y_start:flare_y_end, flare_x_start:flare_x_end]
    result[y_start:y_end, x_start:x_end] += flare_region * intensity
        
    return np.clip(result, 0, 255).astype(np.uint8)

def watercolor_effect(image, intensity=0.3):
    """Эффект акварельной бумаги"""
    # Загружаем текстуру
    texture = cv2.imread("watercolor_paper.jpg")
        
    h, w = image.shape[:2]
        
    # Масштабируем текстуру
    if texture.shape[:2] != (h, w):
        texture = resize_image(texture, new_width=w, new_height=h)
        
    # Преобразуем в float 
    img_float = image.astype(np.float32)
    texture_float = texture.astype(np.float32)
        
    # Создаем маску 
    texture_gray = np.mean(texture_float, axis=2)
    texture_mask = 1 - (texture_gray / 255.0)
        
    # Усиливаем маску
    texture_mask = texture_mask ** (1 / max(0.1, intensity))
    texture_mask = texture_mask[:, :, np.newaxis]
        
    # Смешиваем 
    blended = img_float * (1 - texture_mask * intensity) + texture_float * (texture_mask * intensity)
        
    return np.clip(blended, 0, 255).astype(np.uint8)

def parse_color(color_str):
    """Парсинг строки цвета"""
    try:
        r, g, b = map(int, color_str.split(','))
        return [b, g, r]  
    except:
        raise ValueError(f"Invalid color format: {color_str}. Use R,G,B")

def main():
    args = argument_parser()
    
    try:
        # Загрузка изображения
        image = load_image(args.image)
        result = None
        
        # Применение выбранного фильтра
        if args.filter == 'resize':
            result = resize_image(image, args.scale, args.width, args.height)
            
        elif args.filter == 'sepia':
            result = sepia_filter(image, args.sepia_intensity)
            
        elif args.filter == 'vignette':
            result = vignette_filter(image, args.vignette_strength, args.vignette_radius)
            
        elif args.filter == 'pixelate':
            if args.pixel_x is not None and args.pixel_y is not None and args.pixel_width is not None and args.pixel_height is not None:
                # Пикселизация по заданным координатам
                result = pixelate_region(image, args.pixel_x, args.pixel_y, 
                                       args.pixel_width, args.pixel_height, 
                                       args.pixel_size)
            else:
                # Интерактивная пикселизация
                result = interactive_pixelation(image, args.pixel_size)
                
        elif args.filter == 'border':
            border_color = parse_color(args.border_color)
            result = add_border(image, args.border_width, border_color)
            
        elif args.filter == 'fancy_border':
            border_color = parse_color(args.fancy_border_color)
            result = add_fancy_border(image, args.fancy_border_type, border_color)
            
        elif args.filter == 'lens_flare':
            result = lens_flare_effect(image, args.flare_x, args.flare_y, args.flare_intensity)
    
        elif args.filter == 'watercolor':
            result = watercolor_effect(image, args.watercolor_intensity)
        
        # Отображение и сохранение результата
        show_images(image, result, args.filter)
        save_image(result, args.output)
        print(f"Result saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
