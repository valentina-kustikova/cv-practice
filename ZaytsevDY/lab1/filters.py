import os
import cv2
import numpy as np

# Изменение размера изображения
def resize_image(image, new_width, new_height):
    h, w = image.shape[0], image.shape[1]

    if len(image.shape) == 3:
        resized = np.zeros((new_height, new_width, 3), dtype=image.dtype)
    else:
        resized = np.zeros((new_height, new_width), dtype=image.dtype)

    x_ratio = w / new_width
    y_ratio = h / new_height

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * x_ratio)
            src_y = int(y * y_ratio)

            resized[y, x] = image[src_y, src_x]

    return resized

# Фильтр - сепия
def sepia_filter(image, k):
    height, width = image.shape[:2]
    sepia_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            intensity = 0.299 * r + 0.587 * g + 0.114 * b

            new_r = np.clip(intensity + 2 * k, 0, 255)
            new_g = np.clip(intensity + 0.5 * k, 0, 255)
            new_b = np.clip(intensity - k, 0, 255)

            sepia_image[y, x] = [new_b, new_g, new_r]

    return sepia_image.astype(np.uint8)

# Фильтр - виньетка
def vignette_filter(image, strength):
    output = image.copy().astype(np.float32)

    height, width = image.shape[:2]

    center_x = width // 2
    center_y = height // 2

    max_distance = np.sqrt(center_x**2 + center_y**2)

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            darken = 1 - strength * (distance / max_distance)

            if len(image.shape) == 3:
                output[y, x] = image[y, x] * darken
            else:
                output[y, x] = image[y, x] * darken

    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

# Фильтр - пикселизация
def pixelate_region(image, x, y, width, height, pixel_size):
    result = image.copy()

    region = result[y:y + height, x:x + width]

    h, w = region.shape[:2]

    for i in range(0, h, pixel_size):
        for j in range(0, w, pixel_size):
            end_i = min(i + pixel_size, h)
            end_j = min(j + pixel_size, w)

            block = region[i:end_i, j:end_j]

            if len(block.shape) == 3:
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            else:
                avg_color = np.mean(block).astype(np.uint8)

            region[i:end_i, j:end_j] = avg_color

    return result

# Функция для прямоугольной рамки
def add_border(image, border_width, color):
    result = image.copy()

    height, width = image.shape[:2]

    border_width = max(0, border_width)

    if border_width == 0:
        return result

    for y in range(border_width):
        for x in range(width):
            result[y, x] = color

    for y in range(height - border_width, height):
        for x in range(width):
            result[y, x] = color

    for x in range(border_width):
        for y in range(height):
            result[y, x] = color

    for x in range(width - border_width, width):
        for y in range(height):
            result[y, x] = color

    return result

# Функция для фигурной рамки (теперь только photo_frame)
def add_fancy_border(image, border_width):
    result = image.copy()
    height, width = image.shape[:2]

    border_width = max(0, border_width)
    if border_width == 0:
        return result

    # Только паттерн photo_frame
    frame_path = "photo_frame.png"
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Ошибка загрузки рамки: {frame_path}")
        print("Используется простая черная рамка")
        return add_border(image, border_width, (0, 0, 0))

    frame_height, frame_width = frame.shape[:2]
    photo_height, photo_width = image.shape[:2]

    print(f"Размер рамки: {frame_width}x{frame_height}")
    print(f"Размер фото: {photo_width}x{photo_height}")

    # Используем border_width как процент отступов
    margin_percent = min(border_width, 100)  # Ограничиваем 100%
    margin_x = int(frame_width * margin_percent / 100)
    margin_y = int(frame_height * margin_percent / 100)

    insert_x = margin_x
    insert_y = margin_y
    insert_width = frame_width - 2 * margin_x
    insert_height = frame_height - 2 * margin_y

    if insert_width <= 0 or insert_height <= 0:
        print("Слишком большая рамка для данного изображения")
        print("Используется простая черная рамка")
        return add_border(image, 10, (0, 0, 0))

    print(f"Область для вставки: {insert_width}x{insert_height}")

    photo_resized = resize_image(image, insert_width, insert_height)

    result = frame.copy()

    result[insert_y:insert_y + insert_height, insert_x:insert_x + insert_width] = photo_resized

    return result

# Функция для наложения бликов
def add_lens_flare(image):
    flare_path = "flare.png"

    flare_img = cv2.imread(flare_path)

    if flare_img is None:
        print(f"Файл {flare_path} не найден!")
        return image

    height, width = image.shape[:2]
    flare_resized = resize_image(flare_img, width, height)

    result = image.copy().astype(np.float32)
    flare_float = flare_resized.astype(np.float32)

    for y in range(height):
        for x in range(width):
            b1, g1, r1 = result[y, x]

            b2, g2, r2 = flare_float[y, x]

            new_r = 255 - ((255 - r1) * (255 - r2) / 255)
            new_g = 255 - ((255 - g1) * (255 - g2) / 255)
            new_b = 255 - ((255 - b1) * (255 - b2) / 255)

            new_r = min(255, max(0, new_r))
            new_g = min(255, max(0, new_g))
            new_b = min(255, max(0, new_b))

            result[y, x] = [new_b, new_g, new_r]

    return result.astype(np.uint8)

# Функция для наложения текстуры акварельной бумаги
def add_watercolor_texture(image):
    paper_path = "water.jpg"

    paper_img = cv2.imread(paper_path)

    if paper_img is None:
        print(f"Файл {paper_path} не найден!")
        return image

    height, width = image.shape[:2]
    paper_resized = resize_image(paper_img, width, height)

    result = image.copy()

    for y in range(height):
        for x in range(width):
            paper_b, paper_g, paper_r = paper_resized[y, x]

            img_b, img_g, img_r = result[y, x]

            paper_brightness = (float(paper_b) + float(paper_g) + float(paper_r)) / 765.0

            brightness_factor = 0.6 + 0.4 * paper_brightness

            new_b = min(255, int(img_b * brightness_factor))
            new_g = min(255, int(img_g * brightness_factor))
            new_r = min(255, int(img_r * brightness_factor))

            result[y, x] = [new_b, new_g, new_r]

    return result

def show_menu():
    print("\n" + "=" * 50)
    print("          ФИЛЬТРЫ ДЛЯ ИЗОБРАЖЕНИЙ")
    print("=" * 50)
    print("1. Изменение разрешения")
    print("2. Эффект сепии")
    print("3. Эффект виньетки")
    print("4. Пикселизация области")
    print("5. Прямоугольная рамка")
    print("6. Фигурная рамка (только photo_frame)")
    print("7. Эффект бликов")
    print("8. Текстура акварельной бумаги")
    print("0. Выход")
    print("=" * 50)

def get_user_choice():
    while True:
        try:
            choice = int(input("\nВыберите фильтр (0-8): "))
            if 0 <= choice <= 8:
                return choice
            else:
                print("Пожалуйста, введите число от 0 до 8")
        except ValueError:
            print("Пожалуйста, введите корректное число")

def load_image():
    while True:
        filename = input("Введите путь к изображению: ")
        if os.path.exists(filename):
            image = cv2.imread(filename)
            if image is not None:
                print(f"Изображение загружено: {image.shape[1]}x{image.shape[0]} пикселей")
                return image
            else:
                print("Ошибка загрузки изображения. Попробуйте другой файл.")
        else:
            print("Файл не найден. Попробуйте еще раз.")

def apply_filter(choice, image):
    if choice == 1:
        print("\n--- Изменение разрешения ---")
        new_width = int(input("Новая ширина: "))
        new_height = int(input("Новая высота: "))
        return resize_image(image, new_width, new_height)

    elif choice == 2:
        print("\n--- Эффект сепии ---")
        k = input("Интенсивность эффекта [20]: ")
        k = int(k) if k else 20
        return sepia_filter(image, k)

    elif choice == 3:
        print("\n--- Эффект виньетки ---")
        strength = float(input("Сила эффекта [0.8]: ") or 0.8)
        return vignette_filter(image, strength)

    elif choice == 4:
        print("\n--- Пикселизация области ---")
        x = int(input("Координата X: "))
        y = int(input("Координата Y: "))
        w = int(input("Ширина области: "))
        h = int(input("Высота области: "))
        pixel_size = int(input("Размер пикселя: "))
        return pixelate_region(image, x, y, w, h, pixel_size)

    elif choice == 5:
        print("\n--- Прямоугольная рамка ---")
        width = int(input("Ширина рамки: "))
        print("Цвет рамки (B G R):")
        b = int(input("Синий (0-255): "))
        g = int(input("Зеленый (0-255): "))
        r = int(input("Красный (0-255): "))
        return add_border(image, width, (b, g, r))

    elif choice == 6:
        print("\n--- Фигурная рамка (photo_frame) ---")
        width = int(input("Отступ от краев рамки (в процентах 0-100): "))
        return add_fancy_border(image, width)

    elif choice == 7:
        print("\n--- Эффект бликов ---")
        return add_lens_flare(image)

    elif choice == 8:
        print("\n--- Текстура акварельной бумаги ---")
        return add_watercolor_texture(image)

    return image

def main():
    print("Загрузите изображение для начала работы.")

    image = load_image()

    if image is None:
        print("Не удалось загрузить изображение. Выход.")
        return

    while True:
        show_menu()
        choice = get_user_choice()

        if choice == 0:
            print("Выход из программы. До свидания!")
            break

        try:
            result = apply_filter(choice, image)

            cv2.imshow('Исходное изображение', image)
            cv2.imshow('Результат', result)

            print("\nРезультат показан в окне. Нажмите любую клавишу в окне изображения чтобы продолжить...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            print("Попробуйте еще раз с другими параметрами.")

if __name__ == "__main__":
    main()