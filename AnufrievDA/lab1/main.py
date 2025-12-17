import cv2
import numpy as np
import os
import argparse
import filters

# --- Глобальные переменные для callback-функции ---
drawing = False  # True, если кнопка мыши зажата
ix, iy = -1, -1  # Начальные координаты
x, y = -1, -1    # Текущие координаты

# --- Callback-функция для обработки событий мыши ---
def draw_rectangle(event, current_x, current_y, flags, param):
    global ix, iy, x, y, drawing

    # Событие: кнопка мыши нажата
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = current_x, current_y
        x, y = current_x, current_y # Сбрасываем текущие координаты

    # Событие: мышь двигается
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x, y = current_x, current_y

    # Событие: кнопка мыши отпущена
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x, y = current_x, current_y
        
def main():
    parser = argparse.ArgumentParser(
        description="Применяет выбранный фильтр к изображению и показывает результат.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("filter", help="Название фильтра для применения.",
                        choices=['resize', 'sepia', 'vignette', 'pixelate', 'simple_frame',
                                 'image_frame', 'flare', 'paper'])
    parser.add_argument("input_path", help="Путь к входному изображению.")
    parser.add_argument("--output", type=str, help="Путь для сохранения результата. Если не указан, результат будет только показан.")
    
    parser.add_argument("--width", type=int, help="Новая ширина для фильтра 'resize'.")
    parser.add_argument("--height", type=int, help="Новая высота для фильтра 'resize'.")
    parser.add_argument("--pixel-size", type=int, default=20, help="Размер 'пикселя' для фильтра 'pixelate'.")
    parser.add_argument("--thickness", type=int, default=30, help="Толщина для фильтра 'simple_frame'.")
    parser.add_argument("--color", type=str, default="0,0,255", help="Цвет для 'simple_frame' в формате B,G,R (например, '0,0,255' для красного).")
    parser.add_argument("--center-x", type=int, help="Координата X центра для 'pixelate' или 'flare'.")
    parser.add_argument("--center-y", type=int, help="Координата Y центра для 'pixelate' или 'flare'.")
    parser.add_argument("--roi-size", type=int, help="Размер (ширина и высота) области для 'pixelate'.")
    parser.add_argument("--flare-scale", type=float, default=1.0, help="Масштаб для 'flare'.")
    
    args = parser.parse_args()

    ASSETS_DIR = 'assets'
    FRAME_PATH = os.path.join(ASSETS_DIR, 'border1.png')
    FLARE_PATH = os.path.join(ASSETS_DIR, 'flare.jpg')
    PAPER_PATH = os.path.join(ASSETS_DIR, 'paper.jpg')

    original_image = cv2.imread(args.input_path)
    if original_image is None:
        print(f"Ошибка: не удалось загрузить изображение по пути '{args.input_path}'")
        return

    h, w = original_image.shape[:2]
    result_image = None
    filter_name = args.filter

    try:
        if filter_name == 'pixelate':
            window_name = "Select area: Drag mouse, press ENTER to apply, ESC to cancel"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, draw_rectangle)
            
            roi_x, roi_y, roi_w, roi_h = -1, -1, -1, -1

            while True:
                # Создаем копию, чтобы рисовать на ней, не затирая оригинал
                temp_image = original_image.copy()
                
                # Если пользователь рисует, отображаем прямоугольник
                if drawing or (ix != -1 and x != -1):
                    cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
                
                cv2.imshow(window_name, temp_image)
                
                key = cv2.waitKey(10) & 0xFF
                
                # Нажат ESC - отмена
                if key == 27:
                    result_image = original_image.copy() # Возвращаем оригинал
                    break
                # Нажат Enter - применяем фильтр
                elif key == 13:
                    # Определяем правильные координаты (x, y, w, h)
                    roi_x = min(ix, x)
                    roi_y = min(iy, y)
                    roi_w = abs(ix - x)
                    roi_h = abs(iy - y)
                    
                    if roi_w > 0 and roi_h > 0:
                        result_image = filters.pixelate_area(original_image, x=roi_x, y=roi_y, w=roi_w, h=roi_h, pixel_size=args.pixel_size)
                    else:
                        result_image = original_image.copy()
                    break
            
            cv2.destroyWindow(window_name)

        # ... (остальные вызовы elif без изменений) ...
        elif filter_name == 'pixelate':
            center_x = args.center_x if args.center_x is not None else w // 2
            center_y = args.center_y if args.center_y is not None else h // 2
            roi_size = args.roi_size if args.roi_size is not None else min(w, h) // 2
            roi_x = center_x - roi_size // 2
            roi_y = center_y - roi_size // 2
            result_image = filters.pixelate_area(original_image, x=roi_x, y=roi_y, w=roi_size, h=roi_size, pixel_size=args.pixel_size)
        elif filter_name == 'resize':
            target_width = args.width if args.width is not None else w // 2
            target_height = args.height if args.height is not None else h // 2
            result_image = filters.resize_image(original_image, width=target_width, height=target_height)
        elif filter_name == 'sepia':
            result_image = filters.apply_sepia(original_image)
        elif filter_name == 'vignette':
            result_image = filters.apply_vignette(original_image)
        elif filter_name == 'simple_frame':
            try:
                b, g, r = map(int, args.color.split(','))
                frame_color = (b, g, r)
            except ValueError:
                print("Ошибка: неверный формат цвета. Используйте B,G,R.")
                return
            result_image = filters.add_simple_frame(original_image, thickness=args.thickness, color=frame_color)
        elif filter_name == 'image_frame':
            result_image = filters.add_image_frame(original_image, frame_path=FRAME_PATH)
        elif filter_name == 'flare':
            center_x = args.center_x if args.center_x is not None else w // 2
            center_y = args.center_y if args.center_y is not None else h // 2
            result_image = filters.add_image_flare(original_image, 
                                                   flare_path=FLARE_PATH, 
                                                   center_x=center_x, 
                                                   center_y=center_y, 
                                                   scale=args.flare_scale)
        elif filter_name == 'paper':
            result_image = filters.apply_paper_texture(original_image, paper_path=PAPER_PATH)


    except FileNotFoundError as e:
        print(f"Ошибка: Не найден необходимый файл ассета! {e}")
        return
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return

    if result_image is not None:
        # --- ИСПРАВЛЕННАЯ ЛОГИКА ОТОБРАЖЕНИЯ ---
        
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(args.output, result_image)
            print(f"Результат сохранен в файл: {args.output}")
            
        # Для всех фильтров, кроме 'resize', высота одинакова, просто склеиваем
        if filter_name != 'resize':
            combined_image = np.hstack([original_image, result_image])
        # Для 'resize' используем специальную логику
        else:
            h_orig, w_orig = original_image.shape[:2]
            h_res, w_res = result_image.shape[:2]
            # Увеличиваем результат до высоты оригинала для наглядного сравнения
            result_for_display = resize_image_for_display(result_image, (int(w_res * h_orig / h_res), h_orig))
            # Склеиваем ПОЛНОРАЗМЕРНЫЙ ОРИГИНАЛ и увеличенный результат
            combined_image = np.hstack([original_image, result_for_display])

        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        MAX_DISPLAY_WIDTH = 1600
        display_h, display_w = combined_image.shape[:2]
        if display_w > MAX_DISPLAY_WIDTH:
            scale_ratio = MAX_DISPLAY_WIDTH / display_w
            new_h = int(display_h * scale_ratio)
            display_image = resize_image_for_display(combined_image, (MAX_DISPLAY_WIDTH, new_h))
        else:
            display_image = combined_image
            
        window_title = f"Original (left) vs {filter_name.capitalize()} (right)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, display_image)
        print(f"Показан результат для фильтра '{filter_name}'. Нажмите любую клавишу для выхода.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def resize_image_for_display(image, size):
    """Вспомогательная функция для изменения размера с помощью cv2.resize, 
    используется только для отображения, не для фильтров."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    main()