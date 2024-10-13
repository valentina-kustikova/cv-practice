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
    cv.waitKey(0)  # Ожидание нажатия клавиши
    cv.destroyAllWindows()
    return image

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
def change_resolution(image_path, new_width, new_height, out_image_path):
    image = load_and_show_image(image_path)
    original_height, original_width = image.shape[:2]
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    block_height = original_height // new_height
    block_width = original_width // new_width

    for i in range(0, new_height):
        for j in range(0, new_width):
            block = image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            avg_color = block.mean(axis=(0, 1))  # Средний цвет блока
            resized_image[i, j] = avg_color

    cv.imwrite(out_image_path, resized_image)
    print(f"Изображение с изменённым разрешением сохранено в '{out_image_path}'.")
    show_filtered_image(out_image_path)

# Виньетка
def vignette_filter(image_path, out_image_path, radius):
    image = load_and_show_image(image_path)
    rows, cols = image.shape[:2]
    vignette_image = np.copy(image)
    
    center_x, center_y = cols // 2, rows // 2
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
            vignette_strength = max(0, 1 - distance / radius)
            for c in range(3):
                vignette_image[i, j, c] = int(vignette_image[i, j, c] * vignette_strength)
    
    cv.imwrite(out_image_path, vignette_image)
    print(f"Виньетированное изображение сохранено в '{out_image_path}'.")
    show_filtered_image(out_image_path)

# Пикселизация с выбором области
def select_region(event, x, y, flags, param):
    global ref_point, cropping
    if event == cv.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True
    elif event == cv.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv.imshow("image", image)

def pixelate_image(image_path, pixel_size, out_image_path):
    global image, ref_point, cropping
    ref_point = []
    cropping = False

    image = load_and_show_image(image_path)
    clone = image.copy()
    cv.namedWindow("image")
    cv.setMouseCallback("image", select_region)
    
    while True:
        cv.imshow("image", image)
        key = cv.waitKey(1) & 0xFF
        
        if key == ord("r"):
            image = clone.copy()
        
        elif key == 13:  # Код клавиши Enter
            break

    if len(ref_point) == 2 and ref_point[0] != ref_point[1]:
        roi = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        height, width = roi.shape[:2]
        temp_image = cv.resize(roi, (width // pixel_size, height // pixel_size), interpolation=cv.INTER_LINEAR)
        pixelated_roi = cv.resize(temp_image, (width, height), interpolation=cv.INTER_NEAREST)
        
        image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] = pixelated_roi

        cv.imwrite(out_image_path, image)
        cv.destroyAllWindows()
        print(f"Пикселизированное изображение сохранено в '{out_image_path}'.")
        show_filtered_image(out_image_path)

    else:
        print("Область для пикселизации не выбрана или выбрана некорректно.")
        cv.destroyAllWindows()

# Оригинальные фильтры
def image_mode(image_path, out_image_path):
    image = load_and_show_image(image_path)
    cv.imwrite(out_image_path, image)
    print(f"Оригинальное изображение сохранено в '{out_image_path}'.")
    show_filtered_image(out_image_path)

# Оттенки серого
def grey_color(image_path, out_image_path):
    image = load_and_show_image(image_path)
    height, width, channels = image.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            B, G, R = image[i, j]
            gray_value = int(0.299 * R + 0.587 * G + 0.114 * B)
            gray_image[i, j] = gray_value

    gray_image_colored = cv.merge([gray_image, gray_image, gray_image])
    cv.imwrite(out_image_path, gray_image_colored)
    print(f"Грейскейл изображение сохранено в '{out_image_path}'.")
    show_filtered_image(out_image_path)
    
# Фильтр сепии
def sepia_filter(image_path, out_image_path):
    image = load_and_show_image(image_path)
    height, width, channels = image.shape
    sepia_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            B, G, R = image[i, j]
            new_R = min(255, int(0.393 * R + 0.769 * G + 0.189 * B))
            new_G = min(255, int(0.349 * R + 0.686 * G + 0.168 * B))
            new_B = min(255, int(0.272 * R + 0.534 * G + 0.131 * B))
            sepia_image[i, j] = [new_B, new_G, new_R]
    
    cv.imwrite(out_image_path, sepia_image)
    print(f"Сепия изображение сохранено в '{out_image_path}'.")
    show_filtered_image(out_image_path)

# Главная функция
def main():
    args = cli_argument_parser()

    if args.mode == 'image':
        image_mode(args.image_path, args.out_image_path)
    elif args.mode == 'grey_color':
        grey_color(args.image_path, args.out_image_path)
    elif args.mode == 'change_resolution':
        change_resolution(args.image_path, args.width, args.height, args.out_image_path)
    elif args.mode == 'sepia_filter':
        sepia_filter(args.image_path, args.out_image_path)
    elif args.mode == 'vignette_filter':
        vignette_filter(args.image_path, args.out_image_path, args.radius)
    elif args.mode == 'pixelated':
        pixelate_image(args.image_path, args.pixel_size, args.out_image_path)
    else:
        raise ValueError('Unsupported mode')

    print(f"Фильтр '{args.mode}' успешно применён. Изображение сохранено в '{args.out_image_path}'.")

if __name__ == '__main__':
    sys.exit(main() or 0)
