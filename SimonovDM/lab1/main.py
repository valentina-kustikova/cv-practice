import argparse
import cv2
import numpy as np

def resize_image(image, width = None, height = None, scale_factor = None):
    h, w = image.shape[:2]

    if (width is None) and (height is None) and (scale_factor is None):
        return image.copy()
    if width == w and height == h:
        return image.copy()

    if scale_factor is not None:
        if scale_factor == 1:
            return image.copy()
        new_height = int(h * scale_factor)
        new_width = int(w * scale_factor)
    else:
        if width is None: new_width = w
        else: new_width = width

        if height is None: new_height = h
        else: new_height = height

    scale_h = float(h / new_height)
    scale_w = float(w / new_width)

    y = (np.arange(new_height) * scale_h).astype(int)
    x = (np.arange(new_width) * scale_w).astype(int)
    x_neigh_index, y_neigh_index = np.meshgrid(x, y)

    result = image[y_neigh_index, x_neigh_index]
    return result

def sepia_filter(image, intensity=1.0):
    sepia_img = np.zeros_like(image, np.uint8)

    sepia_img[:, :, 2] = np.clip(0.393 * intensity * image[:, :, 2] + 0.769 * intensity * image[:, :, 1] + 0.189 * intensity * image[:, :, 0], 0, 255)
    sepia_img[:, :, 1] = np.clip(0.349 * intensity * image[:, :, 2] + 0.686 * intensity * image[:, :, 1] + 0.168 * intensity * image[:, :, 0], 0, 255)
    sepia_img[:, :, 0] = np.clip(0.272 * intensity * image[:, :, 2] + 0.534 * intensity * image[:, :, 1] + 0.131 * intensity * image[:, :, 0], 0, 255)

    return sepia_img

def vignette_filter(image, intensity=0.8, radius=0.8):
    def vignette_mask(height, width, intensity, radius):
        y, x = np.ogrid[:height, :width]
        center_x = width // 2
        center_y = height // 2

        max_radius = min(center_x, center_y) * radius
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        mask = np.exp(-((distance / max_radius) ** 2) * intensity)
        mask = mask / np.max(mask)

        return mask

    h, w = image.shape[:2]

    mask = vignette_mask(h, w, intensity, radius)

    image = image * mask[:, :, np.newaxis]
    image = image.astype(np.uint8)

    return image

def pixelation_image(image, rectangle_start, rectangle_end, block_size):
    x1, y1 = rectangle_start
    x2, y2 = rectangle_end

    result = image.copy()

    for i in range(y1, y2, block_size):
        for j in range(x1, x2, block_size):
            block_y2 = min(i + block_size, y2)
            block_x2 = min(j + block_size, x2)
            block = result[i:block_y2, j:block_x2]
            avg_color = np.mean(block, axis=(0, 1), dtype=int)
            result[i:block_y2, j:block_x2] = avg_color

    return result

def frame_add(image, frame_width=None, R=0, G=0, B=0):
    result = image.copy()
    h, w = image.shape[:2]

    if frame_width is None:
        frame_width = int(min(h, w)/10)

    result[0:frame_width] = [B,G,R]
    result[-frame_width:] = [B,G,R]
    result[frame_width:-frame_width,0:frame_width] = [B,G,R]
    result[frame_width:-frame_width,-frame_width:] = [B,G,R]

    return result

def frame_figure_add(image, frame_number = 1):
    frame_files = ["frame/frame1.png", "frame/frame2.png"]

    if frame_number < 1 or frame_number > len(frame_files):
        print(f"Ошибка: номер рамки должен быть от 1 до {len(frame_files)}")
        return image

    frame = cv2.imread(frame_files[frame_number-1])

    if frame is None:
        print(f"Ошибка: не удалось загрузить рамку")
        return image

    h, w = image.shape[:2]

    if frame.shape != image.shape:
        frame = resize_image(frame, width=w, height=h)

    for x in range(w):
        for y in range(h):
            if not np.all(frame[y][x] == [255, 255, 255]):
                image[y][x] = frame[y][x]

    return image

def glare_add(image, flare_center=(0.5, 0.5), intensity=1.0, scale=1.0):
    h_img, w_img = image.shape[:2]

    glare = cv2.imread("glare/glare1.png")
    if glare is None:
        print(f"Ошибка: не удалось загрузить glare/glare1.png")
        return image

    new_w = int(glare.shape[1] * scale)
    new_h = int(glare.shape[0] * scale)
    glare = resize_image(glare, new_w, new_h)

    cx = int(flare_center[0] * w_img)
    cy = int(flare_center[1] * h_img)

    x_start = cx - new_w // 2
    y_start = cy - new_h // 2
    x_end = x_start + new_w
    y_end = y_start + new_h

    img_x_start = max(0, x_start)
    img_y_start = max(0, y_start)
    img_x_end = min(w_img, x_end)
    img_y_end = min(h_img, y_end)

    glare_x_start = max(0, -x_start)
    glare_y_start = max(0, -y_start)
    glare_x_end = glare_x_start + (img_x_end - img_x_start)
    glare_y_end = glare_y_start + (img_y_end - img_y_start)

    img_region = image[img_y_start:img_y_end, img_x_start:img_x_end].astype(np.float32)
    glare_region = glare[glare_y_start:glare_y_end, glare_x_start:glare_x_end].astype(np.float32)

    img_region = img_region + glare_region * intensity

    image[img_y_start:img_y_end, img_x_start:img_x_end] = np.clip(img_region, 0, 255).astype(np.uint8)

    return image

def aqua_texture_add(image, intensity=0.5):
    def bgr_to_gray(image):
        if image.shape[2] < 3:
            raise ValueError("Ожидается цветное изображение с 3 каналами (BGR)")

        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]

        gray = 0.114 * B + 0.587 * G + 0.299 * R
        return gray

    h, w = image.shape[:2]
    aqua = cv2.imread("aqua/aqua1.png")

    if aqua is None:
        print("Ошибка: не удалось загрузить текстуру акварельной бумаги")
        return image

    aqua = resize_image(aqua, w, h)

    image_float = image.astype(np.float32) / 255.0
    aqua_float = aqua.astype(np.float32) / 255.0

    aqua_gray = bgr_to_gray(aqua_float)
    aqua_gray = aqua_gray[:, :, np.newaxis]

    result = image_float * (aqua_gray * intensity)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Путь к изображению')

    subparsers = parser.add_subparsers(dest='filter_type', help='Тип фильтра', required=True)

    resize = subparsers.add_parser('resize', help='Изменение размера')
    resize.add_argument('--width', type=int, help='Новая ширина')
    resize.add_argument('--height', type=int, help='Новая высота')
    resize.add_argument('--scale', type=float, help='Коэффициент масштабирования')

    sepia = subparsers.add_parser('sepia', help='Применеие фотоэффекта сепии')
    sepia.add_argument('--intensity', type=float, default=1.0, help='Интенсивность применения сепии')

    vignette = subparsers.add_parser('vignette', help='Применеие фотоэффекта виньетки')
    vignette.add_argument('--intensity', type=float, default=0.8, help='Интенсивность виньетки')
    vignette.add_argument('--radius', type=float, default=0.8, help='Радиус виньетки')

    pixelation = subparsers.add_parser('pixelation', help='Применеие фотоэффекта виньетки')
    pixelation.add_argument('--start_x', type=int, required=True, help='X-координата начала области пикселизации')
    pixelation.add_argument('--start_y', type=int, required=True, help='Y-координата начала области пикселизации')
    pixelation.add_argument('--end_x', type=int, required=True, help='X-координата конца области пикселизации')
    pixelation.add_argument('--end_y', type=int, required=True, help='Y-координата конца области пикселизации')
    pixelation.add_argument('--block_size', type=int, default=10, help='Размер пиксельного блока (по умолчанию 10)')

    frame = subparsers.add_parser('frame', help='Наложение прямоугольной одноцветной рамки')
    frame.add_argument('--width', type=int, default=None, help='Ширина рамки')
    frame.add_argument('--r', type=int, default=0, help='Красный цвет')
    frame.add_argument('--g', type=int, default=0, help='Зелёный цвет')
    frame.add_argument('--b', type=int, default=0, help='Синий цвет')

    frame_figure = subparsers.add_parser('frame_figure', help='Наложение фигурной рамки')
    frame_figure.add_argument('--frame_number', type=int, default=1, help='Номер рамки')

    glare = subparsers.add_parser('glare', help='Наложение бликов на изображение')
    glare.add_argument('--center_x', type=float, default=0.5, help='X-координата центра блика (0.0-1.0, относительно ширины)')
    glare.add_argument('--center_y', type=float, default=0.5, help='Y-координата центра блика (0.0-1.0, относительно высоты)')
    glare.add_argument('--intensity', type=float, default=1.0, help='Интенсивность блика (рекомендуется 0.1-2.0)')
    glare.add_argument('--scale', type=float, default=1.0, help='Коэффициент масштабирования')

    aqua_texture = subparsers.add_parser('aqua_texture', help='Наложение бликов на изображение')
    aqua_texture.add_argument('--intensity', type=float, default=0.5, help='Интенсивность (рекомендуется 0.1-2.0)')

    parser = parser.parse_args()

    return parser

def main():
    args = parser()
    img = cv2.imread(args.image)

    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {args.image}")
        return

    result = None

    if args.filter_type == 'resize':
        result = resize_image(img, args.width, args.height, args.scale)

    elif args.filter_type == 'sepia':
        result = sepia_filter(img, args.intensity)

    elif args.filter_type == 'vignette':
        result = vignette_filter(img, args.intensity, args.radius)

    elif args.filter_type == 'pixelation':
        start = (args.start_x, args.start_y)
        end = (args.end_x, args.end_y)
        result = pixelation_image(img, start, end, args.block_size)

    elif args.filter_type == 'frame':
        result = frame_add(img, args.width, args.r, args.g, args.b)

    elif args.filter_type == 'frame_figure':
        result = frame_figure_add(img, args.frame_number)

    elif args.filter_type == 'glare':
        center = (args.center_x, args.center_y)
        result = glare_add(img, center, args.intensity, args.scale)

    elif args.filter_type == 'aqua_texture':
        result = aqua_texture_add(img, args.intensity)

    else:
        print(f"Неизвестный фильтр: {args.filter_type}")
        return

    if result is not None:
        cv2.imshow("Оригинал", img)
        cv2.imshow("Результат", result)
        cv2.waitKey(7000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

