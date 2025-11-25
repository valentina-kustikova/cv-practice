import cv2
import numpy as np
import math

drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
pixel_size = 10

def resize_image(image, new_width, new_height, interpolation='nearest'):
    h, w = image.shape[:2]
    resized = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    scale_x = w / new_width
    scale_y = h / new_height
    
    for y in range(new_height):
        for x in range(new_width):
            if interpolation == 'nearest':
                src_x = min(int(x * scale_x), w - 1)
                src_y = min(int(y * scale_y), h - 1)
                resized[y, x] = image[src_y, src_x]
            elif interpolation == 'bilinear':
                src_x = x * scale_x
                src_y = y * scale_y
                
                x1 = int(src_x)
                y1 = int(src_y)
                x2 = min(x1 + 1, w - 1)
                y2 = min(y1 + 1, h - 1)
                
                wx = src_x - x1
                wy = src_y - y1
                
                for c in range(image.shape[2]):
                    a = image[y1, x1, c] * (1 - wx) + image[y1, x2, c] * wx
                    b = image[y2, x1, c] * (1 - wx) + image[y2, x2, c] * wx
                    resized[y, x, c] = a * (1 - wy) + b * wy
    
    return resized

def apply_sepia(image, intensity=1.0):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    
    result = np.zeros_like(image, dtype=np.float32)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = image[y, x].astype(np.float32)
            new_pixel = np.dot(sepia_filter, pixel)
            new_pixel = np.clip(new_pixel * intensity, 0, 255)
            result[y, x] = new_pixel
    
    return result.astype(np.uint8)

def apply_vignette(image, strength=0.8):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    result = image.astype(np.float32).copy()
    
    for y in range(h):
        for x in range(w):
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            norm_distance = min(distance / max_distance, 1.0)
            vignette = 1 - strength * (1 - math.cos(norm_distance * math.pi / 2))
            
            for c in range(image.shape[2]):
                result[y, x, c] = image[y, x, c] * vignette
    
    return np.clip(result, 0, 255).astype(np.uint8)

def pixelate_region(image, x, y, width, height, pixel_size=10):
    result = image.copy()
    
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    width = min(width, image.shape[1] - x)
    height = min(height, image.shape[0] - y)
    
    for block_y in range(y, y + height, pixel_size):
        for block_x in range(x, x + width, pixel_size):
            block_end_y = min(block_y + pixel_size, y + height)
            block_end_x = min(block_x + pixel_size, x + width)
            
            block = image[block_y:block_end_y, block_x:block_end_x]
            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1)).astype(image.dtype)
                result[block_y:block_end_y, block_x:block_end_x] = avg_color
    
    return result

def add_border(image, border_width, color=(0, 0, 0)):
    h, w = image.shape[:2]
    new_h = h + 2 * border_width
    new_w = w + 2 * border_width
    
    bordered = np.full((new_h, new_w, image.shape[2]), color, dtype=image.dtype)
    bordered[border_width:border_width+h, border_width:border_width+w] = image
    
    return bordered

def add_fancy_border(image, border_width=20, color=(0, 0, 0)):
    h, w = image.shape[:2]
    
    new_h = h + 2 * border_width
    new_w = w + 2 * border_width
    bordered = np.full((new_h, new_w, image.shape[2]), color, dtype=image.dtype)
    
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    
    amplitude = border_width // 3
    frequency = 0.1
    
    for x in range(new_w):
        wave_offset_top = int(amplitude * math.sin(x * frequency))
        y_top = border_width + wave_offset_top
        mask[:y_top, x] = 1
        
        wave_offset_bottom = int(amplitude * math.sin(x * frequency + math.pi))
        y_bottom = h + border_width - wave_offset_bottom
        mask[y_bottom:, x] = 1
    
    for y in range(new_h):
        wave_offset_left = int(amplitude * math.sin(y * frequency))
        x_left = border_width + wave_offset_left
        mask[y, :x_left] = 1
        
        wave_offset_right = int(amplitude * math.sin(y * frequency + math.pi))
        x_right = w + border_width - wave_offset_right
        mask[y, x_right:] = 1
    
    for y in range(new_h):
        for x in range(new_w):
            if mask[y, x] == 1:
                bordered[y, x] = color
            else:
                src_y = y - border_width
                src_x = x - border_width
                if 0 <= src_y < h and 0 <= src_x < w:
                    bordered[y, x] = image[src_y, src_x]
    
    return bordered

def add_lens_flare(image, position, size=50, intensity=0.7):
    result = image.astype(np.float32).copy()
    h, w = image.shape[:2]
    
    flares = [
        (size, intensity, position),
        (size*0.7, intensity*0.6, (position[0]-size//2, position[1]-size//2)),
        (size*0.4, intensity*0.4, (position[0]+size//3, position[1]+size//3))
    ]
    
    for flare_size, flare_intensity, flare_pos in flares:
        flare_x = min(max(flare_pos[0], 0), w-1)
        flare_y = min(max(flare_pos[1], 0), h-1)
        
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((x - flare_x)**2 + (y - flare_y)**2)
                if distance < flare_size:
                    flare_value = math.exp(-(distance**2) / (2*(flare_size/3)**2))
                    flare_value *= flare_intensity
                    
                    for c in range(3):
                        result[y, x, c] = min(255, result[y, x, c] + flare_value * 255)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_watercolor_paper(image, texture_intensity=0.3):
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    paper_texture = generate_paper_texture(h, w)
    
    for y in range(h):
        for x in range(w):
            texture_value = paper_texture[y, x]
            for c in range(3):
                blended = result[y, x, c] * (1 - texture_intensity) + \
                         result[y, x, c] * texture_value * texture_intensity
                result[y, x, c] = blended
    
    return np.clip(result, 0, 255).astype(np.uint8)

def generate_paper_texture(height, width):
    texture = np.zeros((height, width))
    
    for octave in range(4):
        scale = 2 ** octave
        amplitude = 1.0 / (scale + 1)
        
        for y in range(height):
            for x in range(width):
                value = math.sin(x * 0.01 * scale + y * 0.01 * scale) * 0.5 + 0.5
                value += math.sin(x * 0.03 * scale) * math.sin(y * 0.03 * scale)
                texture[y, x] += value * amplitude
    
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    return texture

def load_image():
    image_path = input("Введите путь к изображению (или нажмите Enter для использования test_image.jpg): ").strip()
    if not image_path:
        image_path = "test_image.jpg"
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Файл {image_path} не найден. Создаю тестовое изображение...")
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), -1)
        cv2.circle(image, (450, 200), 80, (0, 255, 0), -1)
        cv2.imwrite("test.jpg", image)
        print("Тестовое изображение создано: test.jpg")
    
    return image

def mouse_callback(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, current_image, original_image_copy, pixel_size
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            temp_image = current_image.copy()
            cv2.rectangle(temp_image, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow('Interactive Pixelation', temp_image)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        width = x2 - x1
        height = y2 - y1
        
        if width > 0 and height > 0:
            current_image = pixelate_region(current_image, x1, y1, width, height, pixel_size)
            cv2.imshow('Interactive Pixelation', current_image)

def interactive_pixelation(original_image):
    global current_image, original_image_copy, pixel_size, drawing, ix, iy, fx, fy
    
    drawing = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    pixel_size = 10
    
    original_image_copy = original_image.copy()
    current_image = original_image_copy.copy()
    
    cv2.namedWindow('Interactive Pixelation')
    cv2.setMouseCallback('Interactive Pixelation', mouse_callback)
    
    print("\n=== ИНТЕРАКТИВНАЯ ПИКСЕЛИЗАЦИЯ ===")
    print("Инструкция:")
    print(" - ЛКМ: Выделить область для пикселизации")
    print(" - 'r': Сбросить к оригинальному изображению")
    print(" - '+': Увеличить размер пикселя")
    print(" - '-': Уменьшить размер пикселя") 
    print(" - 'q': Выход")
    print(f"Текущий размер пикселя: {pixel_size}")
    
    while True:
        cv2.imshow('Interactive Pixelation', current_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            current_image = original_image_copy.copy()
            print("Изображение сброшено к оригиналу")
            print(f"Текущий размер пикселя: {pixel_size}")
            
        elif key == ord('+') or key == ord('='):
            pixel_size = min(50, pixel_size + 2)
            print(f"Размер пикселя увеличен до: {pixel_size}")
            
        elif key == ord('-'):
            pixel_size = max(2, pixel_size - 2)
            print(f"Размер пикселя уменьшен до: {pixel_size}")
            
        elif key == ord('q'):
            print("Выход из режима пикселизации")
            break
        
        if cv2.getWindowProperty('Interactive Pixelation', cv2.WND_PROP_VISIBLE) < 1:
            print("Окно закрыто, выход из режима пикселизации")
            break
    
    cv2.destroyAllWindows()

def show_image(window_name, image):
    cv2.imshow(window_name, image)
    print(f"Изображение открыто. Закройте окно '{window_name}' для продолжения...")
    
    while True:
        key = cv2.waitKey(100)
        if key != -1 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window_name)
    except:
        pass

def filter_resize(original_image):
    print("\n--- Изменение размера ---")
    try:
        new_width = int(input("Введите новую ширину: "))
        new_height = int(input("Введите новую высоту: "))
        interpolation = input("Выберите метод интерполяции (nearest/bilinear) [nearest]: ").strip()
        if not interpolation:
            interpolation = 'nearest'
        
        result = resize_image(original_image, new_width, new_height, interpolation)
        show_image('Resized Image', result)
        
    except ValueError:
        print("Ошибка: введите корректные числа")

def filter_sepia(original_image):
    print("\n--- Сепия фильтр ---")
    try:
        intensity = float(input("Введите интенсивность (0.1-2.0) [1.0]: ").strip() or "1.0")
        result = apply_sepia(original_image, intensity)
        show_image('Sepia Image', result)
            
    except ValueError:
        print("Ошибка: введите корректное число")

def filter_vignette(original_image):
    print("\n--- Виньетка ---")
    try:
        strength = float(input("Введите силу эффекта (0.1-1.0) [0.8]: ").strip() or "0.8")
        result = apply_vignette(original_image, strength)
        show_image('Vignette Image', result)
            
    except ValueError:
        print("Ошибка: введите корректное число")

def filter_border(original_image):
    print("\n--- Простая рамка ---")
    try:
        width = int(input("Введите ширину рамки: "))
        print("Введите цвет рамки в формате R G B (0-255)")
        r = int(input("Красный (R): "))
        g = int(input("Зеленый (G): "))
        b = int(input("Синий (B): "))
        
        result = add_border(original_image, width, (b, g, r))
        show_image('Bordered Image', result)
            
    except ValueError:
        print("Ошибка: введите корректные числа")

def filter_fancy_border(original_image):
    print("\n--- Фигурная рамка ---")
    try:
        width = int(input("Введите ширину рамки: "))
        print("Введите цвет рамки в формате R G B (0-255)")
        r = int(input("Красный (R): "))
        g = int(input("Зеленый (G): "))
        b = int(input("Синий (B): "))
        
        result = add_fancy_border(original_image, width, (b, g, r))
        show_image('Fancy Border Image', result)
            
    except ValueError:
        print("Ошибка: введите корректные числа")

def filter_lens_flare(original_image):
    print("\n--- Эффект бликов ---")
    try:
        x = int(input("Введите X координату центра блика: "))
        y = int(input("Введите Y координату центра блика: "))
        size = int(input("Введите размер блика [50]: ").strip() or "50")
        intensity = float(input("Введите интенсивность (0.1-1.0) [0.7]: ").strip() or "0.7")
        
        result = add_lens_flare(original_image, (x, y), size, intensity)
        show_image('Lens Flare Image', result)
            
    except ValueError:
        print("Ошибка: введите корректные числа")

def filter_watercolor(original_image):
    print("\n--- Текстура акварельной бумаги ---")
    try:
        intensity = float(input("Введите интенсивность текстуры (0.1-1.0) [0.3]: ").strip() or "0.3")
        result = apply_watercolor_paper(original_image, intensity)
        show_image('Watercolor Paper Image', result)
            
    except ValueError:
        print("Ошибка: введите корректное число")

def main():
    print("=== ФОТОФИЛЬТРЫ ===")
    
    original_image = load_image()
    
    while True:
        print(f"\n{'='*50}")
        print("ГЛАВНОЕ МЕНЮ")
        print("1. Показать исходное изображение")
        print("2. Изменить размер")
        print("3. Применить сепию")
        print("4. Добавить виньетку")
        print("5. Интерактивная пикселизация")
        print("6. Добавить простую рамку")
        print("7. Добавить фигурную рамку")
        print("8. Добавить эффект бликов")
        print("9. Наложить текстуру акварельной бумаги")
        print("10. Выход")
        
        choice = input("Выберите опцию (1-10): ").strip()
        
        if choice == '1':
            show_image('Original Image', original_image)
            
        elif choice == '2':
            filter_resize(original_image)
            
        elif choice == '3':
            filter_sepia(original_image)
            
        elif choice == '4':
            filter_vignette(original_image)
            
        elif choice == '5':
            interactive_pixelation(original_image)
            
        elif choice == '6':
            filter_border(original_image)
            
        elif choice == '7':
            filter_fancy_border(original_image)
            
        elif choice == '8':
            filter_lens_flare(original_image)
            
        elif choice == '9':
            filter_watercolor(original_image)
            
        elif choice == '10':
            print("До свидания!")
            cv2.destroyAllWindows()
            break
            
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()