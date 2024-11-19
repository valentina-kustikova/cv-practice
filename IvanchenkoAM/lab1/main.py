import cv2
import numpy as np
import argparse
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'image\',\'sepia\', \'grayscale\', \'resize\', \'vignette\'), \'pixelate\'',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output file name',
                        type=str,
                        default='test_out.jpg',
                        dest='out_image_path')
    parser.add_argument('--width',
                        help='New width (in resize mode)',
                        type=int,
                        default=-1,
                        dest='width')
    parser.add_argument('--height',
                        help='New height (in resize mode)',
                        type=int,
                        default=-1,
                        dest='height')
    parser.add_argument('-r', '--radius',
                        help='Radius of vignette',
                        type=float,
                        default=-1,
                        dest='radius')
    args = parser.parse_args()

    return args

def mode_grayscale(img):
    grayscale_filter = np.array([[0.299, 0.587, 0.114]])
    processed_img = cv2.transform(img, grayscale_filter)
    processed_img = np.clip(processed_img, 0, 255)

    return processed_img

def mode_sepia(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
    processed_img = cv2.transform(img, sepia_filter)
    processed_img = np.clip(processed_img, 0, 255)

    return processed_img

def mode_resize(img, new_width, new_height):
     # Calculate the scaling factors
    scale_x = new_width / img.shape[1]
    scale_y = new_height / img.shape[0]
    
    x_ind = np.floor(np.arange(new_width) / scale_x).astype(int)
    y_ind = np.floor(np.arange(new_height) / scale_y).astype(int)

    processed_img = img[y_ind[:, None], x_ind]

    return processed_img


def mode_vignette(img, out_filename, radius):
    rows, cols = img.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, radius)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, radius)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    processed_img = img.copy()

    for i in range(3):  # Apply to each channel
        processed_img[:,:,i] = processed_img[:,:,i] * mask

    return processed_img

rect_start = None
rect_end = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing
    processed_image = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:
            rect_start = (x, y)
            drawing = True
        else:
            drawing = False
            rect_end = (x, y)
            pixelate_area(processed_image, rect_start, rect_end)
            cv2.imshow('processed_image', processed_image)


def pixelate_area(image, start, end):
    x1, y1 = start
    x2, y2 = end
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    pixel_size = 20

    # Копируем выбранную область
    roi = image[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]

    # Размер уменьшенной копии
    small_h = roi_h // pixel_size
    small_w = roi_w // pixel_size

    for i in range(0, small_h):
        for j in range(0, small_w):
            # Определяем область пикселя в оригинальном изображении
            start_y = i * pixel_size
            end_y = start_y + pixel_size
            start_x = j * pixel_size
            end_x = start_x + pixel_size

            # Обрабатываем, чтобы не выходить за границы
            end_y = min(end_y, roi_h)
            end_x = min(end_x, roi_w)

            # Берем среднее значение цвета в области пикселя
            pixel_block = roi[start_y:end_y, start_x:end_x]
            avg_color = pixel_block.mean(axis=(0, 1)).astype(int)
            
            # Заполняем этот блок средним цветом
            roi[start_y:end_y, start_x:end_x] = avg_color

    # Place the pixelated region back into the image
    image[y1:y2, x1:x2] = roi


def mode_pixelate(img):
    processed = img.copy()
    global rect_start, rect_end, drawing
    cv2.namedWindow("original_image")
    cv2.setMouseCallback("original_image", draw_rectangle, param=processed)
    cv2.imshow('original_image', img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows() 
    return processed

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")
    return img

def process_image(img, args):  
    if(args.mode != 'resize' and args.width != -1):
        raise ValueError('Unknown argument \'width\'')
    if(args.mode != 'resize' and args.height != -1):
        raise ValueError('Unknown argument \'height\'')
    if(args.mode != 'vignette' and args.radius != -1):
        raise ValueError('Unknown argument \'radius\'')


    if args.mode == 'grayscale':
        return mode_grayscale(img)
    elif args.mode == 'sepia':
        return mode_sepia(img)
    elif args.mode == 'resize':
        if(args.width < 0 or args.height < 0):
            raise ValueError('Unspecified or invalid target width and height')
        return mode_resize(img, args.width, args.height)
    elif args.mode == 'vignette':
        if(args.radius < 0):
            raise ValueError('Unspecified or invalid radius')
        return mode_vignette(img, args.out_image_path, args.radius)
    elif args.mode == 'pixelate':
        return mode_pixelate(img)
    else:
        raise ValueError('Unsupported mode')

def save_image(img, output_path):
    cv2.imwrite(output_path, img)


def display(img, processed):
    cv2.imshow('Original image', img)
    cv2.imshow('Processed image', processed)
    cv2.waitKey()
    
    cv2.destroyAllWindows()

def main():
    args = cli_argument_parser()
    try:
        img = load_image(args.image_path)    
        processed = process_image(img, args)
        save_image(img, args.out_image_path)
        if(args.mode != 'pixelate'):
            display(img, processed)

    except Exception as e:
        print(e)     


if __name__ == '__main__':
    sys.exit(main() or 0)
