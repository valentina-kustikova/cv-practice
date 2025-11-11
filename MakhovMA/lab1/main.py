import cv2
import argparse
import os
import sys
from filters import *


def main():
    parser = argparse.ArgumentParser(
        description="Применение фильтров к изображению"
    )

    parser.add_argument("image", help="Путь к изображению для обработки")
    parser.add_argument("filter",
                        help="Тип фильтра (resize, sepia, vignette, pixelate, frame, flare, texture)")
    parser.add_argument("params", nargs="*", help="Дополнительные параметры фильтра (опционально)")

    args = parser.parse_args()
    img = cv2.imread(args.image)

    if img is None:
        print("Ошибка: не удалось загрузить изображение.")
        return

    ftype = args.filter.lower()
    params = args.params
    result = None

    # Обработка выбранного фильтра

    # Параметры: [scale]
    if ftype == "resize":
        scale = float(params[0]) if params else 0.5
        result = resize_image(img, scale)
        print(f"Изменение размера: масштаб {scale}")

    # Параметры: нет
    elif ftype == "sepia":
        result = apply_sepia(img)
        print("Применен эффект сепии")

    # Параметры: [strength]
    elif ftype == "vignette":
        strength = float(params[0]) if params else 0.5
        result = apply_vignette(img, strength)
        print(f"Применена виньетка: сила {strength}")

    # Параметры: [pixel_size]
    elif ftype == "pixelate":
        pixel_size = int(params[0]) if params else 10
        print("Выберите область для пикселизации:")
        print("- Зажмите левую кнопку мыши и выделите область")
        print("- Отпустите кнопку для применения эффекта")
        print("- Нажмите Enter для подтверждения или ESC для отмены")
        selector = PixelateSelector(img, pixel_size)
        result = selector.select_region()
        print("Пикселизация применена")

    elif ftype == "frame":
        # Параметры: [толщина] [цвет_R] [цвет_G] [цвет_B] [тип_текстуры] [путь_к_текстуре] [стиль_рамки]

        # Параметры не указаны:
        # thickness = 20, color = (0, 0, 255), texture_type = 'simple', frame_style = 'wave'

        # Указаны только [толщина] [цвет_R] [цвет_G] [цвет_B]:
        # texture_type = 'simple', frame_style = 'wave'

        # Указаны [толщина] [цвет_R] [цвет_G] [цвет_B] texture_type = 'fancy':
        # texture_path = "textures/fancy_frame_texture.jpg", frame_style = 'wave'

        # Остальные варианты дадут ошибку
        thickness = int(params[0]) if params else 20
        color = (int(params[1]), int(params[2]), int(params[3])) if len(params) > 3 else (0, 0, 255)

        texture_type = 'simple'
        texture_path = None
        frame_style = 'wave'

        if len(params) > 4:
            texture_type = params[4]
            if texture_type == 'fancy':
                if len(params) > 5:
                    texture_path = params[5]
                    if len(params) > 6:
                        frame_style = params[6]
                else:
                    # Используем текстуру по умолчанию для fancy
                    texture_path = "textures/fancy_frame_texture.jpg"
                    if not os.path.exists(texture_path):
                        print("Создание текстуры для fancy рамки...")
                        create_sample_textures()

        result = add_frame(img, color, thickness, texture_type, texture_path, frame_style)
        print(f"Добавлена рамка: тип {texture_type}, толщина {thickness}")

    elif ftype == "flare":
        # Параметры: [путь_к_текстуре] [интенсивность]

        # Параметры не указаны:
        # texture_path = "textures/flare_texture.jpg", intensity = 0.7

        # Указано только [путь_к_текстуре]:
        # intensity = 0.7

        # Остальные варианты дадут ошибку
        if not params:
            # Используем текстуру по умолчанию
            texture_path = "textures/flare_texture.jpg"
            if not os.path.exists(texture_path):
                print("Создание текстуры блика по умолчанию...")
                create_sample_textures()
        else:
            texture_path = params[0]

        intensity = float(params[1]) if len(params) > 1 else 0.7
        result = add_lens_flare(img, texture_path, intensity)
        print(f"Добавлены блики: текстура {texture_path}, интенсивность {intensity}")

    # Параметры: [путь_к_текстуре] [интенсивность]

    # Параметры не указаны:
    # texture_path = "textures/paper_texture.jpg", intensity = 0.3

    # Указано только [путь_к_текстуре]:
    # intensity = 0.3

    # Остальные варианты дадут ошибку
    elif ftype == "texture":
        if not params:
            # Используем текстуру по умолчанию
            texture_path = "textures/paper_texture.jpg"
            if not os.path.exists(texture_path):
                print("Создание текстуры бумаги по умолчанию...")
                create_sample_textures()
        else:
            texture_path = params[0]

        intensity = float(params[1]) if len(params) > 1 else 0.3
        result = add_paper_texture(img, texture_path, intensity)
        print(f"Добавлена текстура бумаги: интенсивность {intensity}")

    else:
        print("Ошибка: неизвестный тип фильтра.")
        print("Доступные фильтры: resize, sepia, vignette, pixelate, frame, flare, texture")
        return

    if result is not None:
        # Сохранение результата
        input_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = f"result_{input_name}_{ftype}.jpg"
        cv2.imwrite(output_path, result)
        print(f"Результат сохранен как: {output_path}")

        # Попытка отображения (если поддерживается)
        try:
            cv2.imshow("Original", img)
            cv2.imshow("Result", result)
            print("Нажмите любую клавишу в окне для закрытия...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Отображение окон не поддерживается в данной среде.")
            print(f"Результат сохранен в файл: {output_path}")


if __name__ == "__main__":
    main()
