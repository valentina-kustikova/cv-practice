import cv2
import numpy as np
import matplotlib.pyplot as plt

# Глобальные переменные для обработки мыши
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
selected_region = None

def create_test_image():
    """Создает тестовое изображение"""
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (500, 300), (0, 255, 0), -1)
    cv2.circle(image, (300, 200), 80, (255, 0, 0), -1)
    cv2.putText(image, 'Image', (150, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

def load_png_textures():
    """Загружает PNG текстуры"""
    border = cv2.imread('border.png', cv2.IMREAD_UNCHANGED)
    flare = cv2.imread('flare.png', cv2.IMREAD_UNCHANGED)
    
    # Если не загрузились, создаем простые
    if border is None:
        border = np.zeros((400, 600, 4), dtype=np.uint8)
        border[:, :, :3] = [0, 0, 0]
        border[:, :, 3] = 255
        border[20:380, 20:580, 3] = 0
    
    if flare is None:
        flare = np.zeros((100, 100, 4), dtype=np.uint8)
        center = 50
        for y in range(100):
            for x in range(100):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist < 40:
                    alpha = int(255 * (1 - dist / 40))
                    flare[y, x] = [255, 255, 255, alpha]
    
    return border, flare

def mouse_callback(event, x, y, flags, param):
    """Обработчик мыши для выбора области"""
    global ix, iy, fx, fy, drawing, selected_region
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        selected_region = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        selected_region = (x1, y1, x2, y2)

def pixelate_selected_region_interactive(image, pixel_size=10):
    """Интерактивная пикселизация"""
    global selected_region, ix, iy, fx, fy, drawing
    
    selected_region = None
    display_image = image.copy()
    
    cv2.namedWindow('Выберите область (ENTER - применить, ESC - отмена)')
    cv2.setMouseCallback('Выберите область (ENTER - применить, ESC - отмена)', mouse_callback)
    
    while True:
        temp_image = display_image.copy()
        
        if drawing or selected_region is not None:
            if selected_region is not None:
                x1, y1, x2, y2 = selected_region
            else:
                x1, y1 = ix, iy
                x2, y2 = fx, fy
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1]-1, x2), min(image.shape[0]-1, y2)
            
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Выберите область (ENTER - применить, ESC - отмена)', temp_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER
            if selected_region is not None:
                x1, y1, x2, y2 = selected_region
                width = x2 - x1
                height = y2 - y1
                
                if width > 0 and height > 0:
                    result = pixelate_region(display_image, x1, y1, width, height, pixel_size)
                    cv2.destroyAllWindows()
                    return result
        
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return image

def apply_border_from_png(image, border_png):
    """Накладывает PNG рамку"""
    h, w = image.shape[:2]
    border_resized = cv2.resize(border_png, (w, h))
    
    if border_resized.shape[2] == 4:
        border_rgb = border_resized[:, :, :3]
        border_alpha = border_resized[:, :, 3] / 255.0
        
        result = image.copy()
        for c in range(3):
            result[:, :, c] = (border_rgb[:, :, c] * border_alpha + 
                              result[:, :, c] * (1 - border_alpha)).astype(np.uint8)
        return result
    else:
        return border_resized

def apply_lens_flare_from_png(image, flare_png, position=None):
    """Накладывает PNG блик"""
    h, w = image.shape[:2]
    
    if position is None:
        position = (w // 2, h // 2)
    
    pos_x, pos_y = position
    result = image.copy()
    flare_h, flare_w = flare_png.shape[:2]
    
    for y in range(h):
        for x in range(w):
            flare_y = y - (pos_y - flare_h // 2)
            flare_x = x - (pos_x - flare_w // 2)
            
            if 0 <= flare_y < flare_h and 0 <= flare_x < flare_w:
                flare_pixel = flare_png[flare_y, flare_x]
                
                if len(flare_pixel) == 4:
                    alpha = flare_pixel[3] / 255.0
                    if alpha > 0:
                        for c in range(3):
                            result[y, x, c] = np.clip(
                                flare_pixel[c] * alpha + result[y, x, c] * (1 - alpha),
                                0, 255
                            ).astype(np.uint8)
    
    return result

def pixelate_region(image, x, y, width, height, pixel_size=10):
    """Пикселизирует область"""
    pixelated_image = image.copy()
    
    img_height, img_width = image.shape[:2]
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    if width <= 0 or height <= 0:
        return image
    
    region = image[y:y+height, x:x+width]
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)
    small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    pixelated_region = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    pixelated_image[y:y+height, x:x+width] = pixelated_region
    
    return pixelated_image

def change_resolution(image, scale_factor):
    """Изменяет разрешение"""
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def apply_sepia(image):
    """Применяет сепию"""
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_image = cv2.transform(image, sepia_filter)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def apply_vignette(image, strength=0.8):
    """Применяет виньетку"""
    height, width = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(width, width/3)
    kernel_y = cv2.getGaussianKernel(height, height/3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = 1 - (1 - mask) * strength
    
    vignette_image = image.copy().astype(np.float32)
    for i in range(3):
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask
    
    return np.clip(vignette_image, 0, 255).astype(np.uint8)

def apply_watercolor_texture(image):
    """Применяет акварельную текстуру"""
    height, width = image.shape[:2]
    noise = np.random.normal(0, 15, (height, width, 3))
    textured_image = image.astype(np.float32) + noise
    textured_image = np.clip(textured_image, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(textured_image, (3, 3), 0)

def main():
    # Загружаем изображение
    original_image = cv2.imread('image.jpg')
    if original_image is None:
        original_image = create_test_image()
    
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    border_png, flare_png = load_png_textures()
    
    print("Библиотека фильтров для обработки изображений")
    
    while True:
        print("\nФильтры:")
        print("1. Изменение разрешения")
        print("2. Эффект сепии")
        print("3. Виньетка")
        print("4. Пикселизация области")
        print("5. Рамка из PNG")
        print("6. Блик из PNG")
        print("7. Акварельная текстура")
        print("8. Показать все фильтры")
        print("0. Выход")
        
        choice = input("\nВыбор: ")
        
        if choice == '0':
            break
        
        elif choice == '1':
            try:
                scale = float(input("Коэффициент масштабирования: "))
                result = change_resolution(original_image_rgb, scale)
                cv2.imshow('Изменение разрешения', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Ошибка")
        
        elif choice == '2':
            result = apply_sepia(original_image_rgb)
            cv2.imshow('Эффект сепии', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        elif choice == '3':
            try:
                strength = float(input("Сила эффекта (0.1-1.0): "))
                result = apply_vignette(original_image_rgb, strength)
                cv2.imshow('Виньетка', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Ошибка")
        
        elif choice == '4':
            try:
                pixel_size = int(input("Размер пикселя: "))
                result = pixelate_selected_region_interactive(original_image_rgb, pixel_size)
                cv2.imshow('Пикселизация', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Ошибка")
        
        elif choice == '5':
            result = apply_border_from_png(original_image_rgb, border_png)
            cv2.imshow('Рамка', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        elif choice == '6':
            h, w = original_image_rgb.shape[:2]
            print("1. Центр")
            print("2. Случайно")
            pos_choice = input("Выбор: ")
            
            if pos_choice == '1':
                position = (w // 2, h // 2)
            else:
                position = (np.random.randint(w//4, 3*w//4), 
                           np.random.randint(h//4, 3*h//4))
            
            result = apply_lens_flare_from_png(original_image_rgb, flare_png, position)
            cv2.imshow('Блик', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        elif choice == '7':
            result = apply_watercolor_texture(original_image_rgb)
            cv2.imshow('Акварель', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        elif choice == '8':
            h, w = original_image_rgb.shape[:2]
            
            results = []
            titles = []
            
            results.append(original_image_rgb)
            titles.append("Оригинал")
            
            results.append(change_resolution(original_image_rgb, 0.5))
            titles.append("Изменение разрешения")
            
            results.append(apply_sepia(original_image_rgb))
            titles.append("Сепия")
            
            results.append(apply_vignette(original_image_rgb, 0.6))
            titles.append("Виньетка")
            
            results.append(pixelate_region(original_image_rgb, w//4, h//4, w//2, h//2))
            titles.append("Пикселизация")
            
            results.append(apply_border_from_png(original_image_rgb, border_png))
            titles.append("Рамка")
            
            results.append(apply_lens_flare_from_png(original_image_rgb, flare_png, (w//2, h//2)))
            titles.append("Блик")
            
            results.append(apply_watercolor_texture(original_image_rgb))
            titles.append("Акварель")
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, (img, title) in enumerate(zip(results, titles)):
                if i < len(axes):
                    axes[i].imshow(img)
                    axes[i].set_title(title)
                    axes[i].axis('off')
            
            for i in range(len(results), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        else:
            print("Неверный выбор")

if __name__ == "__main__":
    main()