# main.py

import cv2
import numpy as np
import os
import argparse
import filters

def main():
    # --- Настройка парсера аргументов ---
    parser = argparse.ArgumentParser(
        description="Применяет выбранный фильтр к изображению и показывает результат.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Обязательные аргументы ---
    # Обновляем список доступных фильтров
    parser.add_argument("filter", help="Название фильтра для применения.",
                        choices=['resize', 'sepia', 'vignette', 'pixelate', 'simple_frame',
                                 'image_frame', 'flare', 'paper'])
    parser.add_argument("input_path", help="Путь к входному изображению.")
    
    parser.add_argument("--flare-x", type=int, help="Координата X центра блика.")
    parser.add_argument("--flare-y", type=int, help="Координата Y центра блика.")
    parser.add_argument("--flare-scale", type=float, default=1.0, help="Масштаб блика (например, 0.5 или 1.2).")
    
    args = parser.parse_args()

    # --- Определение путей к ассетам ---
    ASSETS_DIR = 'assets'
    FRAME_PATH = os.path.join(ASSETS_DIR, 'border1.png')
    FLARE_PATH = os.path.join(ASSETS_DIR, 'flare.jpg')
    PAPER_PATH = os.path.join(ASSETS_DIR, 'paper.jpg')

    # --- Основная логика ---
    original_image = cv2.imread(args.input_path)
    if original_image is None:
        print(f"Ошибка: не удалось загрузить изображение по пути '{args.input_path}'")
        return

    h, w = original_image.shape[:2]
    result_image = None
    filter_name = args.filter

    try:
        # --- Вызов фильтров ---
        if filter_name == 'resize':
            result_image = filters.resize_image(original_image, width=w//2, height=h//2)
        elif filter_name == 'sepia':
            result_image = filters.apply_sepia(original_image)
        elif filter_name == 'vignette':
            result_image = filters.apply_vignette(original_image)
        elif filter_name == 'pixelate':
            result_image = filters.pixelate_area(original_image, x=w//4, y=h//4, w=w//2, h=h//2)
        elif filter_name == 'simple_frame':
            result_image = filters.add_simple_frame(original_image, thickness=30, color=(0, 0, 255))
        
        # --- Вызовы новых фильтров ---
        elif filter_name == 'image_frame':
            result_image = filters.add_image_frame(original_image, frame_path=FRAME_PATH)
        elif filter_name == 'flare':
            # Если координаты не заданы, ставим блик по центру
            center_x = args.flare_x if args.flare_x is not None else w // 2
            center_y = args.flare_y if args.flare_y is not None else h // 2
            result_image = filters.add_image_flare(original_image, 
                                                   flare_path=FLARE_PATH, 
                                                   center_x=center_x, 
                                                   center_y=center_y, 
                                                   scale=args.flare_scale)
        elif filter_name == 'paper':
            result_image = filters.apply_paper_texture(original_image, paper_path=PAPER_PATH)

    except FileNotFoundError as e:
        print(f"Ошибка: Не найден необходимый файл ассета!")
        print(e)
        return
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return

    # --- Отображение результата ---
    if result_image is not None:
        h_res, w_res = result_image.shape[:2]
        original_resized = cv2.resize(original_image, (int(w * h_res / h), h_res))
        combined_image = np.hstack([original_resized, result_image])
    
        # --- ИСПРАВЛЕНИЕ ДЛЯ БОЛЬШИХ ЭКРАНОВ ---
        MAX_DISPLAY_WIDTH = 1600
        display_h, display_w = combined_image.shape[:2]
    
        # Если итоговая картинка слишком широкая, пропорционально уменьшаем ее
        if display_w > MAX_DISPLAY_WIDTH:
            scale_ratio = MAX_DISPLAY_WIDTH / display_w
            new_h = int(display_h * scale_ratio)
            display_image = cv2.resize(combined_image, (MAX_DISPLAY_WIDTH, new_h))
        else:
            display_image = combined_image
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
    
        window_title = f"Original vs {filter_name.capitalize()}"
        
        # Создаем окно с возможностью изменения размера
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        
        # Показываем (возможно, уменьшенное) изображение
        cv2.imshow(window_title, display_image)
        
        print(f"Показан результат для фильтра '{filter_name}'. Нажмите любую клавишу для выхода.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()