import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

def resize_image(image_path, width=None, height=None, scale=None, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    original_height, original_width = image.shape[:2]
    print(f"Исходный размер: {original_width}x{original_height}")
    
    if scale is not None:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
    elif width is not None and height is not None:
        new_width = width
        new_height = height
    elif width is not None:
        ratio = width / original_width
        new_width = width
        new_height = int(original_height * ratio)
    elif height is not None:
        ratio = height / original_height
        new_height = height
        new_width = int(original_width * ratio)
    else:
        raise ValueError("Необходимо указать scale, width, height или width и height")
    
    print(f"Новый размер: {new_width}x{new_height}")
    
    resized_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            src_i = int((i / new_height) * original_height)
            src_j = int((j / new_width) * original_width)
            src_i = min(src_i, original_height - 1)
            src_j = min(src_j, original_width - 1)
            resized_image[i, j] = image[src_i, src_j]
    
    if output_path:
        cv2.imwrite(output_path, resized_image)
        print(f"Результат сохранен в: {output_path}")
    
    return resized_image

def manual_resize(image, new_width, new_height):
    original_height, original_width = image.shape[:2]
    resized = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            src_i = int((i / new_height) * original_height)
            src_j = int((j / new_width) * original_width)
            src_i = min(src_i, original_height - 1)
            src_j = min(src_j, original_width - 1)
            resized[i, j] = image[src_i, src_j]
    
    return resized

def apply_sepia(image, intensity=1.0):
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ])
    
    sepia_matrix = sepia_matrix * intensity
    height, width = image.shape[:2]
    sepia_image = np.zeros_like(image, dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            pixel = image[i, j].astype(np.float32)
            new_pixel = np.dot(sepia_matrix, pixel)
            sepia_image[i, j] = np.clip(new_pixel, 0, 255)
    sepia_image = sepia_image.astype(np.uint8)
    
    if intensity < 1.0:
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(height):
            for j in range(width):
                result[i, j] = (sepia_image[i, j] * intensity + 
                               image[i, j] * (1.0 - intensity))
        sepia_image = np.clip(result, 0, 255).astype(np.uint8)
    
    return sepia_image

def apply_vignette(image, intensity=0.8):
    height, width = image.shape[:2]
    
    center_x, center_y = width // 2, height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    mask = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            distance = math.sqrt((j - center_x)**2 + (i - center_y)**2)
            normalized_distance = distance / max_distance
            mask[i, j] = 1.0 - normalized_distance * intensity
    vignette_image = image.copy().astype(np.float32)
    
    for i in range(height):
        for j in range(width):
            for c in range(3):
                vignette_image[i, j, c] = vignette_image[i, j, c] * mask[i, j]
    vignette_image = np.clip(vignette_image, 0, 255).astype(np.uint8)

    return vignette_image

def apply_pixelation(image, x=0, y=0, width=10, height=10, pixel_size=10):
    pixelated_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    region = pixelated_image[y:y+height, x:x+width]
    
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)
    small_region = manual_resize(region, small_width, small_height)
    
    pixelated_region = manual_resize(small_region, width, height)
    
    pixelated_image[y:y+height, x:x+width] = pixelated_region
    return pixelated_image

def apply_frame(image, frame_width=10, frame_color=(255, 255, 255)):
    height, width = image.shape[:2]
    new_height = height + 2 * frame_width
    new_width = width + 2 * frame_width
    
    framed_image = np.full((new_height, new_width, 3), frame_color, dtype=np.uint8)
    
    if frame_width + height <= new_height and frame_width + width <= new_width:
        framed_image[frame_width:frame_width+height, frame_width:frame_width+width] = image
    
    return framed_image

def apply_figure_frame(image, frame_path):
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Не удалось загрузить рамку: {frame_path}")
    
    if frame.shape[:2] != image.shape[:2]:
        frame = manual_resize(frame, image.shape[1], image.shape[0])

    result = image.copy().astype(np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            brightness = np.mean(frame[i, j]) / 255.0
            if brightness > 0.9: 
                alpha = 0.0
            else:  
                alpha = 1.0
            for c in range(3):
                result[i, j, c] = (image[i, j, c] * (1 - alpha) + 
                                  frame[i, j, c] * alpha)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def rgb_to_grayscale(image):
    height, width = image.shape[:2]
    gray = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            gray_value = (0.299 * image[i, j, 2] + 
                         0.587 * image[i, j, 1] + 
                         0.114 * image[i, j, 0])
            gray[i, j] = int(gray_value)
    
    return gray

def apply_lens_flare(image, flare_path, intensity=0.7, position=None):
    flare = cv2.imread(flare_path)
    if flare is None:
        raise ValueError(f"Не удалось загрузить блик: {flare_path}")
    
    flare_alpha = np.zeros((flare.shape[0], flare.shape[1]), dtype=np.float32)
    
    for i in range(flare.shape[0]):
        for j in range(flare.shape[1]):
            brightness = np.mean(flare[i, j]) / 255.0
            if brightness > 0.1: 
                flare_alpha[i, j] = brightness * intensity
            else:
                flare_alpha[i, j] = 0
    
    flare_height, flare_width = flare.shape[:2]
    img_height, img_width = image.shape[:2]
    
    if position is None:
        pos_x = max(0, img_width - flare_width - 50)
        pos_y = 50
    else:
        pos_x, pos_y = position
    
    pos_x = max(0, min(pos_x, img_width - flare_width))
    pos_y = max(0, min(pos_y, img_height - flare_height))
    
    result_image = image.copy().astype(np.float32)
    
    for i in range(flare_height):
        for j in range(flare_width):
            img_i = pos_y + i
            img_j = pos_x + j
            
            if img_i >= img_height or img_j >= img_width:
                continue
                
            alpha = flare_alpha[i, j]
            
            if alpha < 0.01:
                continue
            
            for c in range(3):
                background = result_image[img_i, img_j, c]
                foreground = flare[i, j, c]
                    
                screen_val = 255 - (255 - background) * (255 - foreground) / 255
                result_image[img_i, img_j, c] = background * (1 - alpha) + screen_val * alpha

    
    return np.clip(result_image, 0, 255).astype(np.uint8)

def watercolor_texture(image, intensity=0.3, strength=0.9):
    texture_path = "/Users/taagordeeva/Desktop/cv-practice/GordeevaTS/lab1/watercolor_paper.jpg"
    texture = cv2.imread(texture_path)
    if texture is None:
        raise ValueError(f"Не удалось загрузить текстуру: {texture_path}")
    
    if texture.shape[:2] != image.shape[:2]:
        texture = manual_resize(texture, image.shape[1], image.shape[0])
    
    texture_gray = rgb_to_grayscale(texture)
    texture_mask = 1 - (texture_gray / 255.0)
    
    height, width = texture_mask.shape
    for i in range(height):
        for j in range(width):
            texture_mask[i, j] = texture_mask[i, j] ** (1 / strength)
    
    texture_mask = texture_mask[:, :, np.newaxis]
    
    blended = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                blended[i, j, c] = (image[i, j, c] * (1 - texture_mask[i, j, 0] * intensity) + 
                                  texture[i, j, c] * (texture_mask[i, j, 0] * intensity))
    
    return np.clip(blended, 0, 255).astype(np.uint8)

def show_comparison(original_path, processed_image):
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_rgb)
    ax1.set_title(f'Оригинал: {original.shape[1]}x{original.shape[0]}')
    ax1.axis('off')
    
    ax2.imshow(processed_rgb)
    ax2.set_title(f'Обработанный: {processed_image.shape[1]}x{processed_image.shape[0]}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Обработка изображений')
    parser.add_argument('image_path', help='Путь к исходному изображению')
    
    parser.add_argument('--width', type=int, help='Новая ширина')
    parser.add_argument('--height', type=int, help='Новая высота')
    parser.add_argument('--scale', type=float, help='Коэффициент масштабирования')
    
    parser.add_argument('--sepia', action='store_true', help='Применить эффект сепии')
    parser.add_argument('--sepia_intensity', type=float, default=1.0, 
                       help='Интенсивность эффекта сепии (0.0 - 2.0)')
    
    parser.add_argument('--vignette', action='store_true', help='Применить эффект виньетки')
    parser.add_argument('--vignette_intensity', type=float, default=0.8,
                       help='Интенсивность виньетки (0.0 - 1.0)')
    
    parser.add_argument('--pixelate', action='store_true', help='Применить пикселизацию')
    parser.add_argument('--pixel_x', type=int, help='X координата области пикселизации')
    parser.add_argument('--pixel_y', type=int, help='Y координата области пикселизации')
    parser.add_argument('--pixel_width', type=int, help='Ширина области пикселизации')
    parser.add_argument('--pixel_height', type=int, help='Высота области пикселизации')
    parser.add_argument('--pixel_size', type=int, default=10,
                       help='Размер пикселя (чем больше, тем более пиксельно)')
    
    parser.add_argument('--frame', action='store_true', help='Добавить прямоугольную рамку')
    parser.add_argument('--frame_width', type=int, default=20,
                       help='Ширина рамки в пикселях')
    parser.add_argument('--frame_color', type=str, default='255,255,255',
                       help='Цвет рамки в формате R,G,B (по умолчанию белый)')
    
    parser.add_argument('--figure_frame', action='store_true', help='Добавить фигурную рамку из файла')
    parser.add_argument('--frame_path', type=str, help='Путь к файлу с рамкой (PNG с прозрачностью)')
    
    parser.add_argument('--lens_flare', action='store_true', help='Добавить эффект бликов объектива')
    parser.add_argument('--flare_path', type=str, help='Путь к файлу с бликом (PNG с прозрачностью)')
    parser.add_argument('--flare_intensity', type=float, default=0.7,
                       help='Интенсивность блика (0.0 - 1.0)')
    parser.add_argument('--flare_x', type=int, help='X координата блика')
    parser.add_argument('--flare_y', type=int, help='Y координата блика')
    
    parser.add_argument('--watercolor', action='store_true', help='Применить эффект акварельной бумаги')
    parser.add_argument('--watercolor_intensity', type=float, default=0.3,
                       help='Интенсивность акварельной текстуры (0.0 - 1.0)')
    parser.add_argument('--watercolor_strength', type=float, default=0.9,
                       help='Сила текстуры (0.1 - 2.0)')
    
    #for all
    parser.add_argument('--output', help='Путь для сохранения результата')
    parser.add_argument('--show', action='store_true', help='Показать сравнение изображений')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Ошибка: файл {args.image_path} не существует")
        return
    
    try:
        image = cv2.imread(args.image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {args.image_path}")
        
        result_image = image.copy()
        
        if args.width is not None or args.height is not None or args.scale is not None:
            print("Применяем изменение размера...")
            temp_path = "temp_input.jpg"
            cv2.imwrite(temp_path, result_image)
            result_image = resize_image(
                image_path=temp_path,
                width=args.width,
                height=args.height,
                scale=args.scale,
                output_path=None
            )
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if args.sepia:
            print("Применяем эффект сепии...")
            result_image = apply_sepia(result_image, args.sepia_intensity)
        
        if args.vignette:
            print("Применяем эффект виньетки...")
            result_image = apply_vignette(result_image, args.vignette_intensity)
        
        if args.pixelate:
            if args.pixel_x is not None and args.pixel_y is not None and args.pixel_width is not None and args.pixel_height is not None:
                print(f"Применяем пикселизацию к области ({args.pixel_x}, {args.pixel_y}, {args.pixel_width}, {args.pixel_height})...")
                result_image = apply_pixelation(
                    result_image, 
                    x=args.pixel_x, 
                    y=args.pixel_y, 
                    width=args.pixel_width, 
                    height=args.pixel_height, 
                    pixel_size=args.pixel_size
                )
            else:
                print("Предупреждение: для пикселизации необходимо указать все координаты --pixel_x, --pixel_y, --pixel_width, --pixel_height")
                print("Пикселизация не применена")
        
        if args.frame:
            print(f"Добавляем прямоугольную рамку шириной {args.frame_width} пикселей...")
            try:
                frame_color = tuple(int(x) for x in args.frame_color.split(','))
                if len(frame_color) != 3:
                    raise ValueError
                frame_color = (frame_color[2], frame_color[1], frame_color[0])
            except:
                print("Неверный формат цвета, используем белый по умолчанию")
                frame_color = (255, 255, 255)
            
            result_image = apply_frame(
                result_image, 
                frame_width=args.frame_width,
                frame_color=frame_color
            )
        
        if args.figure_frame:
            if not args.frame_path:
                print("Ошибка: для фигурной рамки необходимо указать --frame_path")
                return
            
            if not os.path.exists(args.frame_path):
                print(f"Ошибка: файл с рамкой {args.frame_path} не существует")
                return
            
            print(f"Добавляем фигурную рамку из файла {args.frame_path}...")
            result_image = apply_figure_frame(
                result_image, 
                frame_path=args.frame_path
            )
        
        if args.lens_flare:
            if not args.flare_path:
                print("Ошибка: для эффекта бликов необходимо указать --flare_path")
                return
            
            if not os.path.exists(args.flare_path):
                print(f"Ошибка: файл с бликом {args.flare_path} не существует")
                return
            
            print(f"Добавляем эффект бликов из файла {args.flare_path}...")
            
            position = None
            if args.flare_x is not None and args.flare_y is not None:
                position = (args.flare_x, args.flare_y)
            
            result_image = apply_lens_flare(
                result_image,
                flare_path=args.flare_path,
                intensity=args.flare_intensity,
                position=position
            )
        
        if args.watercolor:
            print(f"Применяем эффект акварельной бумаги (интенсивность: {args.watercolor_intensity})...")
            result_image = watercolor_texture(
                result_image,
                intensity=args.watercolor_intensity,
                strength=args.watercolor_strength
            )
        
        if args.output and result_image is not None:
            cv2.imwrite(args.output, result_image)
            print(f"Финальный результат сохранен в: {args.output}")
        
        if args.show and result_image is not None:
            show_comparison(args.image_path, result_image)
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()