import argparse

import cv2
import numpy as np


def resize_image(image, new_width, new_height):
    height, width, channels = image.shape
    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    scale_x = width / new_width
    scale_y = height / new_height

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)
            src_x = min(max(src_x, 0), width - 1)
            src_y = min(max(src_y, 0), height - 1)
            resized_image[y, x] = image[src_y, src_x]

    return resized_image


def apply_sepia(image):
    height, width, _ = image.shape
    sepia_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            new_r = int(0.393 * r + 0.769 * g + 0.189 * b)
            new_g = int(0.349 * r + 0.686 * g + 0.168 * b)
            new_b = int(0.272 * r + 0.534 * g + 0.131 * b)
            sepia_image[y, x] = [min(new_b, 255), min(new_g, 255), min(new_r, 255)]
    return sepia_image


def apply_vignette(image, strength=0.5):
    height, width, _ = image.shape
    vignette_image = image.copy()
    center_y, center_x = height // 2, width // 2
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist
            factor = 1 - strength * dist ** 2
            vignette_image[y, x] = np.clip(vignette_image[y, x] * factor, 0, 255)
    return vignette_image


def pixelate_region(image, x, y, w, h, pixel_size=10):
    pixelated = image.copy()
    for i in range(y, y + h, pixel_size):
        for j in range(x, x + w, pixel_size):
            block = pixelated[i:i + pixel_size, j:j + pixel_size]
            mean_color = np.mean(block, axis=(0, 1)).astype(int)
            pixelated[i:i + pixel_size, j:j + pixel_size] = mean_color
    return pixelated


def add_rect_border(image, border_width=10, color=(0, 0, 0)):
    height, width, _ = image.shape
    bordered = image.copy()
    bordered[0:border_width, :] = color
    bordered[height - border_width:height, :] = color
    bordered[:, 0:border_width] = color
    bordered[:, width - border_width:width] = color
    return bordered


def add_shaped_border(image, border_width=10, color=(0, 0, 255), shape_type='wave'):
    height, width, _ = image.shape
    bordered = image.copy()
    if shape_type == 'wave':
        for x in range(width):
            offset = int(border_width * (1 + np.sin(x / 10) / 2))
            bordered[0:offset, x] = color
            bordered[height - offset:height, x] = color
        for y in range(height):
            offset = int(border_width * (1 + np.sin(y / 10) / 2))
            bordered[y, 0:offset] = color
            bordered[y, width - offset:width] = color
    elif shape_type == 'dots':
        dot_size = 8
        dot_spacing = 12
        for x in range(0, width, dot_spacing):
            end_x = min(x + dot_size, width)
            bordered[0:dot_size, x:end_x] = color
        for x in range(0, width, dot_spacing):
            end_x = min(x + dot_size, width)
            bordered[height - dot_size:height, x:end_x] = color
        for y in range(0, height, dot_spacing):
            end_y = min(y + dot_size, height)
            bordered[y:end_y, 0:dot_size] = color
        for y in range(0, height, dot_spacing):
            end_y = min(y + dot_size, height)
            bordered[y:end_y, width - dot_size:width] = color
    return bordered


def add_lens_flare(image, flare_intensity=100):
    height, width, _ = image.shape
    flared = image.copy().astype(float)
    center_y, center_x = height // 4, width // 4
    radius = min(height, width) // 10
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if dist < radius:
                factor = (radius - dist) / radius * flare_intensity
                flared[y, x] += [factor, factor / 2, factor / 3]
    return np.clip(flared, 0, 255).astype(np.uint8)


def add_watercolor_texture(image, texture_path='watercolor_texture.jpg', opacity=0.3):
    texture = cv2.imread(texture_path)
    if texture is None:
        raise ValueError("Текстура не найдена")
    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    textured = (image.astype(float) * (1 - opacity) + texture.astype(float) * opacity).astype(np.uint8)
    return textured


def main():
    parser = argparse.ArgumentParser(description="Применение фильтров к изображению")
    parser.add_argument("image_path", type=str, help="Путь к изображению")
    parser.add_argument("filter_type", type=str,
                        help="Тип фильтра: resize, sepia, vignette, pixelate, rect_border, shaped_border, lens_flare, watercolor")
    parser.add_argument("--params", nargs="*", default=[], help="Параметры фильтра (зависит от типа)")

    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        print("Изображение не найдено")
        return

    filter_type = args.filter_type
    params = args.params

    if filter_type == "resize":  # пример: py lab1_filters.py input_image.jpg resize --params 800 600
        width, height = int(params[0]), int(params[1])
        result = resize_image(image, width, height)

    elif filter_type == "sepia":  # пример: py lab1_filters.py input_image.jpg sepia
        result = apply_sepia(image)

    elif filter_type == "vignette":  # пример: py lab1_filters.py input_image.jpg vignette --params 10
        strength = float(params[0]) if params else 0.5
        result = apply_vignette(image, strength)

    elif filter_type == "pixelate":  # пример: py lab1_filters.py input_image.jpg pixelate --params 350 250 200 200 10
        # где: x=100, y=150, width=200, height=200, pixel_size=15
        x, y, w, h, size = map(int, params[:5]) if len(params) >= 5 else (0, 0, image.shape[1], image.shape[0], 10)
        result = pixelate_region(image, x, y, w, h, size)

    elif filter_type == "rect_border":  # пример: py lab1_filters.py input_image.jpg rect_border --params 20 0 0 255
        # где: ширина=20, цвет=(0, 0, 255) - красный
        width = int(params[0]) if params else 10
        color = tuple(map(int, params[1:4])) if len(params) >= 4 else (0, 0, 255)
        result = add_rect_border(image, width, color)

    elif filter_type == "shaped_border":  # пример: py lab1_filters.py input_image.jpg shaped_border --params 20 0 0 255 wave
        # где: ширина=20, цвет=(0, 0, 255) - красный, wave - форма(wave/dots)
        width = int(params[0]) if params else 10
        color = tuple(map(int, params[1:4])) if len(params) >= 4 else (0, 0, 255)
        shape = params[4] if len(params) >= 5 else 'wave'
        result = add_shaped_border(image, width, color, shape)

    elif filter_type == "lens_flare":  # пример: py lab1_filters.py input_image.jpg lens_flare --params 200
        # где: 200 - интенсивность блика
        intensity = int(params[0]) if params else 100
        result = add_lens_flare(image, intensity)

    elif filter_type == "watercolor":  # пример: py lab1_filters.py input_image.jpg watercolor --params watercolor_texture.jpg 0.4
        texture_path = params[0] if params else 'watercolor_texture.jpg'
        opacity = float(params[1]) if len(params) >= 2 else 0.3
        result = add_watercolor_texture(image, texture_path, opacity)
    else:
        print("Неизвестный фильтр")
        return

    cv2.imshow("Original", image)
    cv2.imshow("Filtered", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_" + filter_type + ".jpg", result)
    print("Результат сохранен как output_" + filter_type + ".jpg")


if __name__ == "__main__":
    main()
