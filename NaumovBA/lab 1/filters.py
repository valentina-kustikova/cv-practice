import cv2
import numpy as np
import math
import os


TEXTURES_DIR = "textur"

DEFAULT_TEXTURES = {
    'flare': "flare_texture.png",   
    'border': "border_texture.png",   
    'paper': "paper_texture.jpg"     
}

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

def get_texture_path(texture_type, user_input=None):

    if user_input is None or user_input.strip() == "":

        filename = DEFAULT_TEXTURES.get(texture_type, "texture.jpg")
        texture_path = os.path.join(TEXTURES_DIR, filename)
    else:
        texture_path = user_input.strip()

        if not os.path.dirname(texture_path):
            texture_path = os.path.join(TEXTURES_DIR, texture_path)
    
    return texture_path

def add_lens_flare_texture(image, position, texture_path=None, intensity=0.7):

    result = image.astype(np.float32).copy()
    h, w = image.shape[:2]

    actual_texture_path = get_texture_path('flare', texture_path)
    
    if not os.path.exists(actual_texture_path):
        print(f"Файл текстуры не найден: {actual_texture_path}")
        return image
    
    flare_texture = cv2.imread(actual_texture_path, cv2.IMREAD_UNCHANGED)
    if flare_texture is None:
        print(f"Не удалось загрузить текстуру: {actual_texture_path}")
        return image
    

    if flare_texture.shape[2] == 3:

        gray = cv2.cvtColor(flare_texture, cv2.COLOR_BGR2GRAY)
        alpha = gray.astype(np.float32) / 255.0
        flare_texture = np.dstack([flare_texture, alpha])
    else:

        flare_texture = flare_texture.astype(np.float32)
        flare_texture[:, :, 3] /= 255.0
    
    flare_h, flare_w = flare_texture.shape[:2]
    flare_x, flare_y = position
    

    for y in range(max(0, flare_y - flare_h//2), min(h, flare_y + flare_h//2)):
        for x in range(max(0, flare_x - flare_w//2), min(w, flare_x + flare_w//2)):

            tex_y = y - (flare_y - flare_h//2)
            tex_x = x - (flare_x - flare_w//2)
            
            if 0 <= tex_y < flare_h and 0 <= tex_x < flare_w:

                tex_color = flare_texture[tex_y, tex_x, :3]
                tex_alpha = flare_texture[tex_y, tex_x, 3] * intensity

                for c in range(3):
                    result[y, x, c] = min(255, result[y, x, c] * (1 - tex_alpha) + tex_color[c] * tex_alpha)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def add_fancy_border_texture(image, border_width, texture_path=None):
    h, w = image.shape[:2]
    
    actual_texture_path = get_texture_path('border', texture_path)

    if not os.path.exists(actual_texture_path):
        print(f"Файл текстуры не найден: {actual_texture_path}")
        print("Проверьте, что в папке 'textur' есть файл 'border_texture.png'")
        return image

    border_texture = cv2.imread(actual_texture_path, cv2.IMREAD_UNCHANGED)
    if border_texture is None:
        print(f"Не удалось загрузить текстуру: {actual_texture_path}")
        return image

    if border_texture.shape[2] == 3:
        print("ВНИМАНИЕ: Текстура не имеет альфа-канала. Создаю простую маску.")

        alpha = np.zeros((border_texture.shape[0], border_texture.shape[1]), dtype=np.uint8)
        
        border_size = min(h, w) // 10 
        alpha[:border_size, :] = 255  
        alpha[-border_size:, :] = 255  
        alpha[:, :border_size] = 255  
        alpha[:, -border_size:] = 255  
        
        
        border_texture = np.dstack([border_texture, alpha])
    
    
    texture_color = border_texture[:, :, :3]
    texture_alpha = border_texture[:, :, 3] / 255.0  
    

    texture_color_resized = cv2.resize(texture_color, (w, h))
    texture_alpha_resized = cv2.resize(texture_alpha, (w, h))
    

    result = image.astype(np.float32).copy()
    
    for y in range(h):
        for x in range(w):
            alpha = texture_alpha_resized[y, x]

            if alpha > 0:
                for c in range(3):

                    result[y, x, c] = result[y, x, c] * (1 - alpha) + texture_color_resized[y, x, c] * alpha
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_watercolor_paper_texture(image, texture_path=None, intensity=0.5):

    h, w = image.shape[:2]
    
    actual_texture_path = get_texture_path('paper', texture_path)
    
    if not os.path.exists(actual_texture_path):
        print(f"Файл текстуры не найден: {actual_texture_path}")
        return image
    
    paper_texture = cv2.imread(actual_texture_path)
    if paper_texture is None:
        print(f"Не удалось загрузить текстуру: {actual_texture_path}")
        return image
    
    if len(paper_texture.shape) == 3:
        gray_texture = cv2.cvtColor(paper_texture, cv2.COLOR_BGR2GRAY)
    else:
        gray_texture = paper_texture
    
    if gray_texture.shape[0] != h or gray_texture.shape[1] != w:
        gray_texture = cv2.resize(gray_texture, (w, h))
    
    texture_norm = gray_texture.astype(np.float32) / 255.0
    result = image.astype(np.float32).copy()
    
    for y in range(h):
        for x in range(w):
            tex_val = texture_norm[y, x]
            blend_val = 1.0 - intensity + tex_val * intensity
            result[y, x] *= blend_val
    
    return np.clip(result, 0, 255).astype(np.uint8)

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



def filter_lens_flare_texture(original_image):
    print("\n--- Эффект бликов ---")
    
    try:
        x = int(input("Введите X координату центра блика: "))
        y = int(input("Введите Y координату центра блика: "))
        
        texture_input = input("Введите имя файла текстуры или нажмите Enter для текстуры по умолчанию: ")
        
        intensity = float(input("Введите интенсивность (0.1-1.0) [0.7]: ").strip() or "0.7")

        result = add_lens_flare_texture(original_image, (x, y), texture_input, intensity)
        show_image('Lens Flare (Texture)', result)
            
    except ValueError:
        print("Ошибка: введите корректные числа")
    except Exception as e:
        print(f"Ошибка при применении эффекта: {e}")

def filter_fancy_border_texture(original_image):
    print("\n--- Фигурная рамка ---")
    try:
        texture_input = input("Введите имя файла текстуры (например, 'border.png') или нажмите Enter для текстуры по умолчанию: ")

        result = add_fancy_border_texture(original_image, 20, texture_input)
        show_image('Fancy Border (Texture Overlay)', result)
            
    except ValueError:
        print("Ошибка: введите корректные числа")
    except Exception as e:
        print(f"Ошибка при применении эффекта: {e}")
        import traceback
        traceback.print_exc()

def filter_watercolor_texture(original_image):
    print("\n--- Текстура акварельной бумаги ---")
  
    try:

        texture_input = input("Введите имя файла текстуры или нажмите Enter для текстуры по умолчанию: ")
        
        intensity = float(input("Введите интенсивность (0.1-1.0) [0.5]: ").strip() or "0.5")


        result = apply_watercolor_paper_texture(original_image, texture_input, intensity)
        show_image('Watercolor Paper (Texture)', result)
            
    except ValueError:
        print("Ошибка: введите корректное число")
    except Exception as e:
        print(f"Ошибка при применении эффекта: {e}")


def main():

    if not os.path.exists(TEXTURES_DIR):
        print(f"\nВНИМАНИЕ: Папка '{TEXTURES_DIR}' не найдена!")
        print("Создайте папку 'textur' и поместите туда текстуры:")
        print("  - flare_texture.png (для бликов)")
        print("  - border_texture.jpg (для рамки)")
        print("  - paper_texture.jpg (для бумаги)")
        print("Или укажите полный путь к текстурам при запросе.\n")
    
    original_image = load_image()
    
    while True:
        print(f"\n{'='*60}")
        print("ГЛАВНОЕ МЕНЮ")
        print("\n=== ОСНОВНЫЕ ФИЛЬТРЫ (процедурные) ===")
        print(" 1. Показать исходное изображение")
        print(" 2. Изменить размер")
        print(" 3. Применить сепию")
        print(" 4. Добавить виньетку")
        print(" 5. Интерактивная пикселизация")
        print(" 6. Добавить простую рамку")
        
        print("\n=== ФИЛЬТРЫ С ТЕКСТУРАМИ ИЗ ФАЙЛОВ ===")
        print(" 7. Добавить эффект бликов (текстурный)")
        print(" 8. Добавить фигурную рамку (текстурную)")
        print(" 9. Наложить текстуру акварельной бумаги (из файла)")
        print(" 10. Выход")
        print('='*60)
        
        choice = input("\nВыберите опцию (1-10): ").strip()
        
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
            filter_lens_flare_texture(original_image)
        elif choice == '8':
            filter_fancy_border_texture(original_image)
        elif choice == '9':
            filter_watercolor_texture(original_image)
        elif choice == '10':
            print("До свидания!")
            cv2.destroyAllWindows()
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()