import cv2
import sys
import os
from filters import *


def save_result(filtered, filename, result_dir='result'):
    # Создаем папку result, если её нет
    os.makedirs(result_dir, exist_ok=True)
    
    # Формируем путь для сохранения
    output_path = os.path.join(result_dir, filename)
    
    # Сохраняем изображение
    cv2.imwrite(output_path, filtered)
    print(f"Результат сохранён: {output_path}")


def demonstrate_filters(image_path):
    # Загружаем исходное изображение
    original = cv2.imread(image_path)
    
    if original is None:
        print(f"Ошибка: не удалось загрузить изображение '{image_path}'")
        return
    
    print(f"Изображение успешно загружено: {image_path}")
    print(f"Размер: {original.shape[1]}x{original.shape[0]}")
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ФИЛЬТРОВ")
    print("="*60)
    
    # Сохраняем оригинал
    save_result(original, "00_original.jpg")
    
    # 1. Изменение разрешения
    print("\n1. Фильтр изменения разрешения")
    resized = resize_image(original, scale=0.5)
    save_result(resized, "01_resize.jpg")
    
    # 2. Эффект сепии
    print("\n2. Фильтр эффекта сепии")
    sepia = apply_sepia(original, intensity=0.9)
    save_result(sepia, "02_sepia.jpg")
    
    # 3. Эффект виньетки
    print("\n3. Фильтр виньетки")
    vignette = apply_vignette(original, strength=0.7)
    save_result(vignette, "03_vignette.jpg")
    
    # 4. Пикселизация области
    print("\n4. Фильтр пикселизации области")
    h, w = original.shape[:2]
    pixelated = pixelate_region(original, 
                                x=w//4, y=h//4, 
                                width=w//2, height=h//2, 
                                pixel_size=20)
    save_result(pixelated, "04_pixelate.jpg")
    
    # 5. Простая рамка
    print("\n5. Фильтр простой рамки")
    simple_frame = add_simple_frame(original, frame_width=30, color=(0, 0, 255))
    save_result(simple_frame, "05_simple_frame.jpg")
    
    # 6. Фигурная рамка - волны
    print("\n6. Фильтр фигурной рамки (волны)")
    decorative_waves = add_decorative_frame(original, frame_width=40, 
                                           color=(255, 215, 0), pattern='waves')
    save_result(decorative_waves, "06_decorative_frame_waves.jpg")
    
    # 6b. Фигурная рамка - зигзаг
    print("\n6b. Фильтр фигурной рамки (зигзаг)")
    decorative_zigzag = add_decorative_frame(original, frame_width=35, 
                                            color=(0, 255, 0), pattern='zigzag')
    save_result(decorative_zigzag, "06b_decorative_frame_zigzag.jpg")
    
    # 6c. Фигурная рамка - круги
    print("\n6c. Фильтр фигурной рамки (круги)")
    decorative_circles = add_decorative_frame(original, frame_width=30, 
                                             color=(255, 0, 255), pattern='circles')
    save_result(decorative_circles, "06c_decorative_frame_circles.jpg")
    
    # 7. Эффект бликов
    print("\n7. Фильтр эффекта бликов объектива")
    lens_flare = add_lens_flare(original, intensity=0.8)
    save_result(lens_flare, "07_lens_flare.jpg")
    
    # 8. Текстура акварельной бумаги
    print("\n8. Фильтр текстуры акварельной бумаги")
    watercolor = add_watercolor_texture(original, intensity=0.6)
    save_result(watercolor, "08_watercolor.jpg")
    
    # Комбинированный пример
    print("\n9. БОНУС: Комбинация фильтров")
    print("   (сепия + виньетка + простая рамка)")
    combined = apply_sepia(original, intensity=0.8)
    combined = apply_vignette(combined, strength=0.5)
    combined = add_simple_frame(combined, frame_width=25, color=(139, 69, 19))
    save_result(combined, "09_combined.jpg")
    
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("Все результаты сохранены в папке 'result'")
    print("="*60)


def apply_single_filter(image_path, filter_name, **kwargs):
    """
    Применяет один фильтр к изображению с заданными параметрами
    """
    # Загружаем изображение
    original = cv2.imread(image_path)
    
    if original is None:
        print(f"Ошибка: не удалось загрузить изображение '{image_path}'")
        return
    
    # Применяем фильтр
    if filter_name == 'resize':
        filtered = resize_image(original, **kwargs)
    elif filter_name == 'sepia':
        filtered = apply_sepia(original, **kwargs)
    elif filter_name == 'vignette':
        filtered = apply_vignette(original, **kwargs)
    elif filter_name == 'pixelate':
        filtered = pixelate_region(original, **kwargs)
    elif filter_name == 'pixelate_interactive':
        filtered = interactive_pixelate(original, **kwargs)
    elif filter_name == 'simple_frame':
        filtered = add_simple_frame(original, **kwargs)
    elif filter_name == 'decorative_frame':
        filtered = add_decorative_frame(original, **kwargs)
    elif filter_name == 'lens_flare':
        filtered = add_lens_flare(original, **kwargs)
    elif filter_name == 'watercolor':
        filtered = add_watercolor_texture(original, **kwargs)
    else:
        print(f"Ошибка: неизвестный фильтр '{filter_name}'")
        print(f"Доступные фильтры: {', '.join(FILTERS.keys())}")
        return
    
    # Формируем имя выходного файла
    input_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_{filter_name}{ext}"
    
    # Сохраняем результат
    save_result(filtered, output_filename)
    print(f"Фильтр '{filter_name}' успешно применён")


def print_usage():
    print("""
Использование:
    python main.py <путь_к_изображению> [фильтр] [параметры]

Режимы работы:
    1. Демонстрация всех фильтров:
       python main.py <путь_к_изображению>
    
    2. Применение конкретного фильтра:
       python main.py <путь_к_изображению> <название_фильтра> [параметры]

Доступные фильтры:
    resize          - изменение разрешения
    sepia           - эффект сепии
    vignette        - эффект виньетки
    pixelate             - пикселизация области (по координатам)
    pixelate_interactive - интерактивная пикселизация (выбор мышью)
    simple_frame    - простая рамка
    decorative_frame - фигурная рамка
    lens_flare      - эффект бликов
    watercolor      - текстура акварели

Примеры:
    # Показать все фильтры
    python main.py image.jpg
    
    # Применить сепию с интенсивностью 0.8
    python main.py image.jpg sepia --intensity 0.8
    
    # Изменить размер с масштабом 0.5
    python main.py image.jpg resize --scale 0.5
    
    # Добавить виньетку
    python main.py image.jpg vignette --strength 0.7
    
    # Добавить фигурную рамку
    python main.py image.jpg decorative_frame --pattern waves --frame_width 40
""")


def main():
    """
    Главная функция скрипта
    """
    if len(sys.argv) < 2:
        print("Ошибка: не указан путь к изображению")
        print_usage()
        return
    
    image_path = sys.argv[1]
    
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"Ошибка: файл '{image_path}' не найден")
        return
    
    # Если указан только путь к изображению - демонстрируем все фильтры
    if len(sys.argv) == 2:
        demonstrate_filters(image_path)
    # Если указан фильтр - применяем его с параметрами
    elif len(sys.argv) >= 3:
        filter_name = sys.argv[2]
        
        # Парсим параметры командной строки
        kwargs = {}
        i = 3
        while i < len(sys.argv):
            if sys.argv[i].startswith('--'):
                param_name = sys.argv[i][2:]
                if i + 1 < len(sys.argv):
                    param_value = sys.argv[i + 1]
                    # Пытаемся преобразовать в число
                    try:
                        if '.' in param_value:
                            param_value = float(param_value)
                        else:
                            param_value = int(param_value)
                    except ValueError:
                        pass  # Оставляем как строку
                    kwargs[param_name] = param_value
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        apply_single_filter(image_path, filter_name, **kwargs)


if __name__ == "__main__":
    main()
