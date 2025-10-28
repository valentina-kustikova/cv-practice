#!/usr/bin/env python3
"""
Демонстрационный скрипт для показа работы всех фильтров.
"""

import cv2 as cv
import numpy as np
import os
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from script import (
    resize_image, sepia_filter, vignette_filter, pixelate_region,
    add_rectangular_frame, add_decorative_frame, add_lens_flare,
    add_watercolor_texture
)

def demo_all_filters():
    """Демонстрирует работу всех фильтров."""
    print("Загрузка тестового изображения из файла...")
    image_path = "images/test_image_1.jpg"
    test_image = cv.imread(image_path)
    if test_image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        print("Создание тестового изображения...")
        test_image = create_test_image()
    
    # Создаем папку для результатов
    os.makedirs('demo_results', exist_ok=True)
    
    print("Демонстрация фильтров:")
    
    # 1. Изменение разрешения
    print("1. Изменение разрешения...")
    resized = resize_image(test_image, 600, 400)
    cv.imwrite('demo_results/1_resize.jpg', resized)
    
    # 2. Эффект сепии
    print("2. Эффект сепии...")
    sepia = sepia_filter(test_image)
    cv.imwrite('demo_results/2_sepia.jpg', sepia)
    
    # 3. Эффект виньетки
    print("3. Эффект виньетки...")
    vignette = vignette_filter(test_image, 0.7)
    cv.imwrite('demo_results/3_vignette.jpg', vignette)
    
    # 4. Обычная пикселизация
    print("4. Пикселизация области...")
    pixelated = pixelate_region(test_image, 50, 50, 150, 100, 15)
    cv.imwrite('demo_results/4_pixelate.jpg', pixelated)
    
    # 4.1 Пикселизация с коллбэком (эффект шахматной доски)
    print("4.1. Пикселизация с коллбэком...")
    
    def chess_pattern_callback(block, bx, by):
        """Коллбэк, создающий шахматный узор на пикселизированной области."""
        if (bx + by) % 2 == 0:
            return block
        else:
            # Инвертируем цвета блока для создания шахматного эффекта
            return 255 - block
    
    pixelated_with_callback = pixelate_region(test_image, 50, 50, 150, 100, 15, chess_pattern_callback)
    cv.imwrite('demo_results/4_1_pixelate_with_callback.jpg', pixelated_with_callback)
    
    # 5. Прямоугольная рамка
    print("5. Прямоугольная рамка...")
    framed = add_rectangular_frame(test_image, 20, (0, 255, 0))
    cv.imwrite('demo_results/5_frame.jpg', framed)
    
    # 6. Декоративная рамка (волнистая)
    print("6. Декоративная рамка (волнистая)...")
    decorative = add_decorative_frame(test_image, "wavy", 25, (255, 0, 0))
    cv.imwrite('demo_results/6_decorative_wavy.jpg', decorative)
    
    # 7. Декоративная рамка (узорная)
    print("7. Декоративная рамка (узорная)...")
    decorative_pattern = add_decorative_frame(test_image, "pattern", 25, (0, 0, 255))
    cv.imwrite('demo_results/7_decorative_pattern.jpg', decorative_pattern)
    
    # 7.1. Декоративная рамка (прямая)
    print("7.1. Декоративная рамка (прямая)...")
    decorative_straight = add_decorative_frame(test_image, "straight", 25, (255, 0, 255))
    cv.imwrite('demo_results/7_1_decorative_straight.jpg', decorative_straight)
    
    # 7.2. Декоративная рамка (красная)
    print("7.2. Декоративная рамка (красная)...")
    decorative_red = add_decorative_frame(test_image, "red", 25, (0, 255, 255))
    cv.imwrite('demo_results/7_2_decorative_red.jpg', decorative_red)
    
    # 7.3. Декоративная рамка (цветочная)
    print("7.3. Декоративная рамка (цветочная)...")
    decorative_floral = add_decorative_frame(test_image, "floral", 25, (255, 255, 0))
    cv.imwrite('demo_results/7_3_decorative_floral.jpg', decorative_floral)
    
    # 8. Эффект бликов
    print("8. Эффект бликов...")
    flared = add_lens_flare(test_image, 200, 150, 0.8)
    cv.imwrite('demo_results/8_flare.jpg', flared)
    
    # 9. Текстура акварельной бумаги
    print("9. Текстура акварельной бумаги...")
    watercolor = add_watercolor_texture(test_image, 0.8)
    cv.imwrite('demo_results/9_watercolor.jpg', watercolor)
    
    print("\nВсе демонстрационные изображения сохранены в папке 'demo_results/'")
    print("Исходное изображение сохранено как 'demo_results/0_original.jpg'")
    cv.imwrite('demo_results/0_original.jpg', test_image)
    
    # Показываем исходное изображение
    cv.imshow('Исходное изображение', test_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_help():
    """Показывает справку по использованию."""
    print("""
Демонстрационный скрипт для библиотеки фильтров изображений

Использование:
    python demo.py              - запустить демонстрацию всех фильтров
    python demo.py --help       - показать эту справку

Демонстрация создаст папку 'demo_results' с примерами работы всех фильтров:
- 0_original.jpg - исходное изображение
- 1_resize.jpg - изменение разрешения
- 2_sepia.jpg - эффект сепии
- 3_vignette.jpg - эффект виньетки
- 4_pixelate.jpg - пикселизация области
- 4_1_pixelate_with_callback.jpg - пикселизация с эффектом шахматной доски (пример коллбэка)
- 5_frame.jpg - прямоугольная рамка
- 6_decorative_wavy.jpg - волнистая декоративная рамка
- 7_decorative_pattern.jpg - узорная декоративная рамка
- 7_1_decorative_straight.jpg - прямая декоративная рамка
- 7_2_decorative_red.jpg - красная декоративная рамка
- 7_3_decorative_floral.jpg - цветочная декоративная рамка
- 8_flare.jpg - эффект бликов
- 9_watercolor.jpg - текстура акварельной бумаги

Для работы с собственными изображениями используйте:
    python script.py -i <путь_к_изображению> -f <тип_фильтра> [параметры]
""")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        demo_all_filters()
