import argparse
import sys
import cv2 as cv
import numpy as np

# Загрузка и отображение изображения в отдельной функции
def load_and_show_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    cv.imshow("Loaded Image", image)
    out = image.copy()
    cv.waitKey(0)  # Ожидание нажатия клавиши
    cv.destroyAllWindows()
    return out

# Отображение отфильтрованного изображения
def show_filtered_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Filtered image not found or unable to load.")
    
    cv.imshow("Filtered Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='output.jpg',
                        dest='out_image_path')
    parser.add_argument('-m', '--mode',
                        help='Mode (image, grey_color, change_resolution, sepia_filter, vignette_filter, pixelated)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-w', '--width',
                        help='New width for resizing',
                        type=int,
                        default=200,
                        dest='width')
    parser.add_argument('-hg', '--height',
                        help='New height for resizing',
                        type=int,
                        default=200,
                        dest='height')
    parser.add_argument('-px', '--pixel_size',
                        help='Size of pixels for pixelation',
                        type=int,
                        default=5,
                        dest='pixel_size')
    parser.add_argument('-r', '--radius',
                        help='Radius for vignette filter',
                        type=int,
                        default=100,
                        dest='radius')

    args = parser.parse_args()
    return args

# Изменение разрешения с использованием блоков
def change_resolution(image, new_width, new_height, out_image_path):
    #image = load_and_show_image(image_path)
    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    block_height = original_height // new_height
    block_width = original_width // new_width

    for i in range(0, new_height):
        for j in range(0, new_width):
            block = image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            avg_color = block.mean(axis=(0, 1))  # Средний цвет блока
            resized_image[i, j] = avg_color

    print(f"Изображение с изменённым разрешением сохранено в '{out_image_path}'.")
    return out_image_path, resized_image

# Виньетка
def vignette_filter(image, out_image_path, radius):
    rows, cols = image.shape[:2]

    # Создаем маску виньетки на основе расстояний от центра
    center_x, center_y = cols // 2, rows // 2
    y_indices, x_indices = np.indices((rows, cols))
    distances = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    vignette_mask = np.clip(1 - distances / radius, 0, 1)

    B, G, R = cv.split(image)
    B = (B * vignette_mask).astype(np.uint8)
    G = (G * vignette_mask).astype(np.uint8)
    R = (R * vignette_mask).astype(np.uint8)
    
    vignette_image = cv.merge([B, G, R])
    print(f"Виньетированное изображение сохранено в '{out_image_path}'.")
    return out_image_path, vignette_image

# Пикселизация с выбором области
def select_region(image):
    global ref_point, cropping
    ref_point = []
    cropping = False
    clone = image.copy()
    cv.namedWindow("image")
    
    def mouse_callback(event, x, y, flags, param):
        global ref_point, cropping
        if event == cv.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
            cropping = True
        elif event == cv.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cropping = False
            cv.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv.imshow("image", image)
    
    cv.setMouseCallback("image", mouse_callback)
    
    while True:
        cv.imshow("image", image)
        key = cv.waitKey(1) & 0xFF
        if key == ord("r"):
            image = clone.copy()
        elif key == 13:
            break
    
    cv.destroyAllWindows()
    if len(ref_point) == 2 and ref_point[0] != ref_point[1]:
        return ref_point
    else:
        print("Область для пикселизации не выбрана или выбрана некорректно.")
        return None

#Применение пикселизации к выбранной области
def apply_pixelation(image, region, pixel_size, out_image_path):
    x_start, y_start = region[0]
    x_end, y_end = region[1]
    
    # Извлечение выбранной области
    roi = image[y_start:y_end, x_start:x_end]
    height, width = roi.shape[:2]
    
    temp_image = cv.resize(roi, (width // pixel_size, height // pixel_size), interpolation=cv.INTER_LINEAR)
    pixelated_roi = cv.resize(temp_image, (width, height), interpolation=cv.INTER_NEAREST)
    
    # Вставка пикселизированной области обратно в изображение
    image[y_start:y_end, x_start:x_end] = pixelated_roi
    cv.imwrite(out_image_path, image)
    
    print(f"Пикселизированное изображение сохранено в '{out_image_path}'.")
    return out_image_path, image

# Функция пикселизации
def pixelate_image(image_in, pixel_size, out_image_path):
    region = select_region(image_in)
    if region:
        return apply_pixelation(image_in, region, pixel_size, out_image_path)
    else:
        return None, image_in

# Оригинальные фильтры
def image_mode(image, out_image_path):
    print(f"Оригинальное изображение сохранено в '{out_image_path}'.")
    return out_image_path, image

# Оттенки серого
def grey_color(image, out_image_path):
    B, G, R = cv.split(image)  # Разделение изображения на каналы
    gray_image = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)  # Расчет серого оттенка
    gray_image_colored = cv.merge([gray_image, gray_image, gray_image])  # Объединение в 3-канальное изображение
    cv.imwrite(out_image_path, gray_image_colored)
    print(f"Грейскейл изображение сохранено в '{out_image_path}'.")
    return out_image_path, gray_image_colored
    
# Фильтр сепии
def sepia_filter(image, out_image_path):
    B, G, R = cv.split(image)
    sepia_R = (0.393 * R + 0.769 * G + 0.189 * B).clip(0, 255).astype(np.uint8)
    sepia_G = (0.349 * R + 0.686 * G + 0.168 * B).clip(0, 255).astype(np.uint8)
    sepia_B = (0.272 * R + 0.534 * G + 0.131 * B).clip(0, 255).astype(np.uint8)
    sepia_image = cv.merge([sepia_B, sepia_G, sepia_R])
    print(f"Сепия изображение сохранено в '{out_image_path}'.")
    return out_image_path, sepia_image


# Главная функция
def main():
    args = cli_argument_parser()
    image_path = load_and_show_image(args.image_path)
    if args.mode == 'image':
        out_image_path, image_out = image_mode(image_path, args.out_image_path)
    elif args.mode == 'grey_color':
        #grey_color(args.image_path, args.out_image_path)
        out_image_path, image_out = grey_color(image_path, args.out_image_path)
    elif args.mode == 'change_resolution':
        out_image_path, image_out = change_resolution(image_path, args.width, args.height, args.out_image_path)
    elif args.mode == 'sepia_filter':
        out_image_path, image_out = sepia_filter(image_path, args.out_image_path)
    elif args.mode == 'vignette_filter':
        out_image_path, image_out = vignette_filter(image_path, args.out_image_path, args.radius)
    elif args.mode == 'pixelated':
        out_image_path, image_out = pixelate_image(image_path, args.pixel_size, args.out_image_path)
    else:
        raise ValueError('Unsupported mode')
    
    cv.imwrite(out_image_path, image_out)
    show_filtered_image(out_image_path)
    print(f"Фильтр '{args.mode}' успешно применён. Изображение сохранено в '{args.out_image_path}'.")

if __name__ == '__main__':
    sys.exit(main() or 0)
