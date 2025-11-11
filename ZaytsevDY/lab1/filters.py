import os

import cv2
import numpy as np
import sys
import math

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

def add_fancy_border(image, border_width, color, pattern):
    result = image.copy()
    height, width = image.shape[:2]

    border_width = max(0, border_width)
    if border_width == 0:
        return result

    if pattern == "solid":
        return add_border(image, border_width, color)

    elif pattern == "dashed":
        dash_length = min(15, border_width * 3)
        gap_length = min(10, border_width * 2)

        for y in range(border_width):
            x = 0
            while x < width:
                # Рисуем штрих
                for i in range(min(dash_length, width - x)):
                    if x + i < width:
                        result[y, x + i] = color
                x += dash_length + gap_length

        for y in range(height - border_width, height):
            x = 0
            while x < width:
                for i in range(min(dash_length, width - x)):
                    if x + i < width:
                        result[y, x + i] = color
                x += dash_length + gap_length

        for x in range(border_width):
            y = 0
            while y < height:
                for i in range(min(dash_length, height - y)):
                    if y + i < height:
                        result[y + i, x] = color
                y += dash_length + gap_length

        for x in range(width - border_width, width):
            y = 0
            while y < height:
                for i in range(min(dash_length, height - y)):
                    if y + i < height:
                        result[y + i, x] = color
                y += dash_length + gap_length

    elif pattern == "double":
        inner_offset = max(1, border_width // 3)

        result = add_border(result, border_width, color)

        inner_border_width = max(1, border_width // 2)
        inner_y_start = border_width + inner_offset
        inner_y_end = height - border_width - inner_offset
        inner_x_start = border_width + inner_offset
        inner_x_end = width - border_width - inner_offset

        for y in range(inner_y_start, inner_y_start + inner_border_width):
            for x in range(inner_x_start, inner_x_end):
                result[y, x] = color

        for y in range(inner_y_end - inner_border_width, inner_y_end):
            for x in range(inner_x_start, inner_x_end):
                result[y, x] = color

        for x in range(inner_x_start, inner_x_start + inner_border_width):
            for y in range(inner_y_start, inner_y_end):
                result[y, x] = color

        for x in range(inner_x_end - inner_border_width, inner_x_end):
            for y in range(inner_y_start, inner_y_end):
                result[y, x] = color

    elif pattern == "wave":
        amplitude = max(2, border_width // 2)
        frequency = 0.1

        for y in range(border_width):
            for x in range(width):
                wave_offset = int(amplitude * math.sin(x * frequency))
                if y == wave_offset % border_width:
                    result[y, x] = color

        for y in range(height - border_width, height):
            for x in range(width):
                wave_offset = int(amplitude * math.sin(x * frequency))
                if (height - y - 1) == wave_offset % border_width:
                    result[y, x] = color

        for x in range(border_width):
            for y in range(height):
                wave_offset = int(amplitude * math.sin(y * frequency))
                if x == wave_offset % border_width:
                    result[y, x] = color

        for x in range(width - border_width, width):
            for y in range(height):
                wave_offset = int(amplitude * math.sin(y * frequency))
                if (width - x - 1) == wave_offset % border_width:
                    result[y, x] = color

    else:
        print(f"Неизвестный паттерн: {pattern}. Используется solid.")
        return add_border(image, border_width, color)

    return result

def add_lens_flare(image, brightness):
    result = image.copy().astype(np.float32)
    height, width = image.shape[:2]

    num_flares = int(brightness * 5) + 1
    max_radius = min(width, height) // 15

    for _ in range(num_flares):
        flare_x = np.random.randint(0, width)
        flare_y = np.random.randint(0, height)

        radius = np.random.randint(max_radius // 3, max_radius)
        intensity = brightness * np.random.uniform(0.5, 1.5)

        for y in range(max(0, flare_y - radius), min(height, flare_y + radius + 1)):
            for x in range(max(0, flare_x - radius), min(width, flare_x + radius + 1)):
                dx = x - flare_x
                dy = y - flare_y
                distance = math.sqrt(dx*dx + dy*dy)

                if distance <= radius:
                    falloff = 1.0 - (distance / radius)
                    falloff = falloff * falloff

                    glow = falloff * intensity * 100

                    if len(image.shape) == 3:
                        for c in range(3):
                            result[y, x, c] = min(255, result[y, x, c] + glow)
                    else:
                        result[y, x] = min(255, result[y, x] + glow)

    return np.clip(result, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, intensity):
    result = image.copy().astype(np.float32)
    height, width = image.shape[:2]

    noise = np.random.randn(height, width).astype(np.float32)

    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255

    blurred = np.zeros_like(noise, dtype=np.float32)
    k = 3

    for y in range(height):
        for x in range(width):
            s = 0.0
            c = 0
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    yy = min(height - 1, max(0, y + dy))
                    xx = min(width - 1, max(0, x + dx))
                    s += noise[yy, xx]
                    c += 1
            blurred[y, x] = s / c

    if len(image.shape) == 3:
        texture = np.dstack([blurred] * 3)
    else:
        texture = blurred

    strength = intensity * 0.4
    result = result * (1 - strength) + texture * strength

    return np.clip(result, 0, 255).astype(np.uint8)


def show_menu():
    print("\n" + "=" * 50)
    print("          ФИЛЬТРЫ ДЛЯ ИЗОБРАЖЕНИЙ")
    print("=" * 50)
    print("1. Изменение разрешения")
    print("2. Эффект сепии")
    print("3. Эффект виньетки")
    print("4. Пикселизация области")
    print("5. Прямоугольная рамка")
    print("6. Фигурная рамка")
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
        print("\n--- Фигурная рамка ---")
        width = int(input("Ширина рамки: "))
        print("Цвет рамки (B G R):")
        b = int(input("Синий (0-255): "))
        g = int(input("Зеленый (0-255): "))
        r = int(input("Красный (0-255): "))
        print("Доступные узоры: solid, dashed, double, wave")
        pattern = input("Тип узора: ")
        return add_fancy_border(image, width, (b, g, r), pattern)

    elif choice == 7:
        # Эффект бликов
        print("\n--- Эффект бликов ---")
        brightness = float(input("Яркость (например, 1.2 для +20%): "))
        return add_lens_flare(image, brightness)

    elif choice == 8:
        # Текстура акварельной бумаги
        print("\n--- Текстура акварельной бумаги ---")
        intensity = float(input("Интенсивность текстуры: "))
        return add_watercolor_texture(image, intensity)

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