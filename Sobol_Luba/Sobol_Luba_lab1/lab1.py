import cv2 as cv
import argparse
import sys
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')
    parser.add_argument('-m', '--mode', 
                        help='Mode (image, grey_color,resize_image,sepia_filter,vignette_filter,pixelated)',
                        type=str,
                        default='image',
                        dest='mode')
    parser.add_argument('-w', '--width',
                        help='Size (width)',
                        type=str,
                        default='200',
                        dest='width',)
    parser.add_argument('-hg', '--height',
                        help='Size (height)',                   
                        type=str,
                        default='200',
                        dest='height',)
    parser.add_argument('-px', '--pixel_size',
                        help='size pf pixel to pixelated',                   
                        type=str,
                        default='5',
                        dest='pixel_size',)
    parser.add_argument('-r', '--radius',
                        help='radius vignette',                   
                        type=str,
                        default='10',
                        dest='radius',)
    args = parser.parse_args()

    return args


def load_image(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')

    # Загрузка изображения
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Изображение не загружено. Проверьте путь к файлу.")

    return image

def highgui_samples(in_filename, out_filename):
    img = cv.imread(in_filename)
    cv.imwrite(out_filename, img)
    
    cv.imshow('Init image', img)
    cv.waitKey()
    
    cv.destroyAllWindows()

def grey_color(image):
    blue_channel = image[:, :, 0] 
    green_channel = image[:, :, 1] 
    red_channel = image[:, :, 2] 
    
    grey_values = (blue_channel * 0.114 + green_channel * 0.587 + red_channel * 0.299).astype(np.uint8) 
    
    grey_image = np.stack((grey_values, grey_values, grey_values), axis=-1)

    return grey_image

def resize_image(image, new_width, new_height):
    height, width, _ = image.shape
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            orig_x = int(j * width / new_width)
            orig_y = int(i * height / new_height)

            resized_image[i, j] = image[orig_y, orig_x]

    return resized_image

def sepia_filter(image):
    blue_channel = image[:, :, 0] 
    green_channel = image[:, :, 1] 
    red_channel = image[:, :, 2] 
    
    sepia_blue = np.clip(0.272 * red_channel + 0.534 * green_channel + 0.131 * blue_channel, 0, 255).astype(np.uint8) 
    sepia_green = np.clip(0.349 * red_channel + 0.686 * green_channel + 0.168 * blue_channel, 0, 255).astype(np.uint8) 
    sepia_red = np.clip(0.393 * red_channel + 0.769 * green_channel + 0.189 * blue_channel, 0, 255).astype(np.uint8) 
     
    sepia_image = np.stack((sepia_blue, sepia_green, sepia_red), axis=-1)

    return sepia_image


def vignette_filter(image, radius):
    height, width, channels = image.shape

    center_x = width // 2
    center_y = height // 2

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # Вычислите фактор виньетирования
    factor = np.ones_like(distances, dtype=np.float32)
    factor[distances > radius] = (radius / distances[distances > radius])
    factor = np.clip(factor, 0.1, 1.0)  # Не даем фактору стать слишком маленьким

    # Примените фактор виньетирования к каждому каналу
    vignette_image = np.zeros_like(image)
    for channel in range(channels):
        vignette_image[:, :, channel] = np.clip(image[:, :, channel] * factor, 0, 255).astype(np.uint8)

    return vignette_image


x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False

def select_area(event, x, y, flags, image):
    global x1, y1, x2, y2, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y

        # Ограничиваем координаты выбранной области, чтобы они не выходили за пределы изображения
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

def pixelate_area(image, pixel_size):
    pixelated_image = np.copy(image)

    # Создаем окно для отображения изображения
    cv.namedWindow('image')
    cv.setMouseCallback('image', select_area, image)

    while True:
        temp_image = np.copy(image)
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.imshow('image', temp_image)
        key = cv.waitKey(1) & 0xFF
        
        if key == 13:  # Нажмите 'Enter' для пикселизации выбранной области
            if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                for i in range(y1, y2, pixel_size):
                    for j in range(x1, x2, pixel_size):
                        # Определяем усреднённый цвет для блока
                        block = image[i:i+pixel_size, j:j+pixel_size]
                        avg_color = block.mean(axis=(0, 1)).astype(int)

                        # Закрашиваем блок усреднённым цветом
                        pixelated_image[i:i+pixel_size, j:j+pixel_size] = avg_color

                # Объединяем изначальное изображение и пикселизированную область
                combined_image = np.copy(image)
                combined_image[y1:y2, x1:x2] = pixelated_image[y1:y2, x1:x2]

                cv.destroyAllWindows()
                break
        
        elif key == 27:  # Нажмите 'Esc' для выхода
            break

    return combined_image

def showImg(img, new_img, named):
    col=np.vstack([img, new_img])
    cv.imshow(named,col)
    cv.waitKey(0)

def main():
    args = cli_argument_parser()
    image = load_image(args.image_path)

    if args.mode == 'image':
        highgui_samples(args.image_path, args.out_image_path)
    elif args.mode == 'grey_color':
        grey_image = grey_color(image)
        showImg(image, grey_image, 'grey_img')
    elif args.mode == 'resize_image':
        resized_image = resize_image(image, int(args.width), int(args.height))
        cv.imshow('resize_image', resized_image)
        cv.waitKey(0)
    elif args.mode == 'sepia_filter':
        sepia_image = sepia_filter(image)
        showImg(image, sepia_image, 'sepia_img')
    elif args.mode == 'vignette_filter':
        vignette_image = vignette_filter(image, int(args.radius))
        showImg(image, vignette_image, 'vignette_image')
    elif args.mode == 'pixelated':
        pixelate_image = pixelate_area(image, int(args.pixel_size))
        showImg(image, pixelate_image, 'pixelate_image')
    else:
        raise ValueError('Unsupported mode')


if __name__ == '__main__':
    sys.exit(main() or 0)