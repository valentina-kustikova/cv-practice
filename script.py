import cv2
import argparse
import sys
import os
import time
from filters import *

FILTERS = {
    '1': ('resize', 'Изменение размера'),
    '2': ('sepia', 'Сепия'),
    '3': ('vignette', 'Виньетка'),
    '4': ('pixelate', 'Пикселизация'),
    '5': ('frame', 'Рамка'),
    '6': ('decorative', 'Декоративная рамка'),
    '7': ('flare', 'Блик'),
    '8': ('watercolor', 'Акварель'),
    '0': ('exit', 'Выход')
}

def show_menu():
    print("\n" + "="*50)
    print("Выберите фильтр:")
    print("="*50)
    for key, (_, name) in FILTERS.items():
        if key != '0':
            print(f"{key}. {name}")
    print("0. Выход")
    print("="*50)

def apply_filter(image, filter_name, h, w):
    """Применяет выбранный фильтр к изображению"""
    try:
        if filter_name == 'resize':
            width = int(input("Введите ширину (по умолчанию 800): ") or "800")
            height = int(input("Введите высоту (по умолчанию 600): ") or "600")
            return resize_image(image, width, height)
        
        elif filter_name == 'sepia':
            return apply_sepia(image)
        
        elif filter_name == 'vignette':
            strength = float(input("Введите силу эффекта 0.0-1.0 (по умолчанию 0.8): ") or "0.8")
            return apply_vignette(image, strength)
        
        elif filter_name == 'pixelate':
            x = int(input(f"Введите X координату (по умолчанию 100, макс {w}): ") or "100")
            y = int(input(f"Введите Y координату (по умолчанию 100, макс {h}): ") or "100")
            pixel_size = int(input("Введите размер пикселя (по умолчанию 10): ") or "10")
            x2 = min(x + 200, w)
            y2 = min(y + 200, h)
            return pixelize_region(image, x, y, x2, y2, pixel_size)
        
        elif filter_name == 'frame':
            frame_width = int(input("Введите ширину рамки (по умолчанию 20): ") or "20")
            return apply_solid_border(image, frame_width)
        
        elif filter_name == 'decorative':
            frame_width = int(input("Введите ширину рамки (по умолчанию 20): ") or "20")
            border_type = input("Введите тип рамки (dashed, по умолчанию): ") or "dashed"
            return apply_custom_border(image, frame_width, border_type)
        
        elif filter_name == 'flare':
            flare_x = int(input(f"Введите X координату блика (по умолчанию 100, макс {w}): ") or "100")
            flare_y = int(input(f"Введите Y координату блика (по умолчанию 100, макс {h}): ") or "100")
            return apply_lens_flare(image, (flare_x, flare_y))
        
        elif filter_name == 'watercolor':
            strength = float(input("Введите силу эффекта 0.0-1.0 (по умолчанию 0.8): ") or "0.8")
            return apply_watercolor_texture(image, strength)
        
        else:
            raise ValueError(f"Неизвестный фильтр: {filter_name}")
    except Exception as e:
        print(f"Ошибка при применении фильтра: {e}")
        return None

def interactive_mode(image_path):
    """Интерактивный режим с меню"""
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        sys.exit(1)
    
    image = cv2.imread(image_path)
    if image is None:
        print("Ошибка: не удалось загрузить изображение")
        sys.exit(1)
    
    original_image = image.copy()
    h, w = image.shape[:2]
    print(f"Изображение загружено: ({h}, {w}, {image.shape[2] if len(image.shape) > 2 else 1})")
    
    output_counter = 1
    
    while True:
        show_menu()
        choice = input("Ваш выбор: ").strip()
        
        if choice == '0':
            print("Выход из программы.")
            break
        
        if choice not in FILTERS:
            print("Неверный выбор. Попробуйте снова.")
            continue
        
        filter_name, filter_desc = FILTERS[choice]
        
        print(f"\nПрименяется фильтр: {filter_desc}")
        result = apply_filter(original_image, filter_name, h, w)
        
        if result is None:
            continue
        
        output_file = f"result_{output_counter}.jpg"
        cv2.imwrite(output_file, result)
        print(f"Результат сохранен: {output_file}")
        output_counter += 1
        
        cv2.imshow('Исходное изображение', original_image)
        cv2.imshow('Отфильтрованное изображение', result)
        print("Нажмите любую клавишу в окне изображения для закрытия окон...")
        sys.stdout.flush()
        
        # Ждем нажатия клавиши (блокирующий вызов)
        cv2.waitKey(0)
        
        # Закрываем все окна
        cv2.destroyAllWindows()
        # Даем время окнам закрыться
        cv2.waitKey(1)
        time.sleep(0.2)
        
        # Очищаем буфер вывода
        sys.stdout.flush()
        
        # Явно запрашиваем подтверждение для возврата в меню
        print("\nОкна закрыты. Нажмите Enter для возврата в меню...")
        sys.stdout.flush()
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход из программы.")
            return
        
        print()  # Пустая строка для читаемости

def main():
    parser = argparse.ArgumentParser(description='Практическая работа №1: фильтры изображений')
    parser.add_argument('-i', '--image', help='Путь к входному изображению')
    parser.add_argument('-f', '--filter', choices=[
        'resize', 'sepia', 'vignette', 'pixelate', 'frame', 'decorative', 'flare', 'watercolor'
    ], help='Тип фильтра (необязательно, если не указан - интерактивный режим)')
    parser.add_argument('-o', '--output', default='output.jpg', help='Путь к выходному файлу')
    
    parser.add_argument('--width', type=int, default=800, help='Ширина для resize')
    parser.add_argument('--height', type=int, default=600, help='Высота для resize')
    parser.add_argument('--strength', type=float, default=0.8, help='Сила эффекта (vignette, watercolor)')
    parser.add_argument('--x', type=int, default=100, help='X координата для pixelate')
    parser.add_argument('--y', type=int, default=100, help='Y координата для pixelate')
    parser.add_argument('--pixel_size', type=int, default=10, help='Размер пикселя для пикселизации')
    parser.add_argument('--frame_width', type=int, default=20, help='Ширина рамки')
    parser.add_argument('--border_type', type=str, default='dashed', help='Тип фигурной рамки')
    parser.add_argument('--flare_x', type=int, default=100, help='X координата блика')
    parser.add_argument('--flare_y', type=int, default=100, help='Y координата блика')
    
    args = parser.parse_args()
    
    # Если указано только изображение без фильтра - интерактивный режим
    if args.image and not args.filter:
        interactive_mode(args.image)
        return
    
    # Если не указано изображение - ошибка
    if not args.image:
        print("Ошибка: необходимо указать путь к изображению через -i или --image")
        sys.exit(1)
    
    # Режим командной строки (как раньше)
    if not os.path.exists(args.image):
        print(f"Ошибка: файл {args.image} не найден")
        sys.exit(1)
    
    image = cv2.imread(args.image)
    if image is None:
        print("Ошибка: не удалось загрузить изображение")
        sys.exit(1)
    
    h, w = image.shape[:2]
    print(f"Изображение загружено: ({h}, {w}, {image.shape[2] if len(image.shape) > 2 else 1})")
    
    try:
        if args.filter == 'resize':
            result = resize_image(image, args.width, args.height)
        elif args.filter == 'sepia':
            result = apply_sepia(image)
        elif args.filter == 'vignette':
            result = apply_vignette(image, args.strength)
        elif args.filter == 'pixelate':
            x2 = min(args.x + 200, w)
            y2 = min(args.y + 200, h)
            result = pixelize_region(image, args.x, args.y, x2, y2, args.pixel_size)
        elif args.filter == 'frame':
            result = apply_solid_border(image, args.frame_width)
        elif args.filter == 'decorative':
            result = apply_custom_border(image, args.frame_width, args.border_type)
        elif args.filter == 'flare':
            result = apply_lens_flare(image, (args.flare_x, args.flare_y))
        elif args.filter == 'watercolor':
            result = apply_watercolor_texture(image, args.strength)
        else:
            raise ValueError(f"Неизвестный фильтр: {args.filter}")
        
        cv2.imwrite(args.output, result)
        print(f"Результат сохранен: {args.output}")
        
        cv2.imshow('Исходное изображение', image)
        cv2.imshow('Отфильтрованное изображение', result)
        print("Нажмите любую клавишу для закрытия окон...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()