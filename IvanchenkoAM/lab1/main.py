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


def highgui_samples(in_filename, out_filename):
    img = cv2.imread(in_filename)
    cv2.imwrite(out_filename, img)
    
    cv2.imshow('Init image', img)
    cv2.waitKey()
    
    cv2.destroyAllWindows()

def mode_grayscale(in_filename, out_filename):
    img = cv2.imread(in_filename)

    grayscale_filter = np.array([[0.299, 0.587, 0.114]])
    processed_img = cv2.transform(img, grayscale_filter)
    processed_img = np.clip(processed_img, 0, 255)

    cv2.imwrite(out_filename, processed_img)

    cv2.imshow('Original image', img)
    cv2.imshow('Grayscale image', processed_img)
    cv2.waitKey()
    
    cv2.destroyAllWindows()

def mode_sepia(in_filename, out_filename):
    img = cv2.imread(in_filename)

    sepia_filter = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
    processed_img = cv2.transform(img, sepia_filter)
    processed_img = np.clip(processed_img, 0, 255)

    cv2.imwrite(out_filename, processed_img)

    cv2.imshow('Original image', img)
    cv2.imshow('Sepia image', processed_img)
    cv2.waitKey()
    
    cv2.destroyAllWindows()

def mode_resize(in_filename, out_filename, new_width, new_height):
    img = cv2.imread(in_filename)

     # Calculate the scaling factors
    scale_x = new_width / img.shape[1]
    scale_y = new_height / img.shape[0]
    
    # Create the transformation matrix
    transformation_matrix = np.array([[scale_x, 0, 0],
                                       [0, scale_y, 0]])
    
    # Apply the transformation
    processed_img = cv2.warpAffine(img, transformation_matrix, (new_width, new_height))

    cv2.imwrite(out_filename, processed_img)

    cv2.imshow('Original image', img)
    cv2.imshow('Resized image', processed_img)
    cv2.waitKey()
    
    cv2.destroyAllWindows()


def mode_vignette(in_filename, out_filename, radius):
    img = cv2.imread(in_filename)

    rows, cols = img.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, radius)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, radius)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    processed_img = img.copy()

    for i in range(3):  # Apply to each channel
        processed_img[:,:,i] = processed_img[:,:,i] * mask

    cv2.imwrite(out_filename, processed_img)

    cv2.imshow('Original image', img)
    cv2.imshow('Resized image', processed_img)
    cv2.waitKey()
    
    cv2.destroyAllWindows()

rect_start = None
rect_end = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing == False:
            rect_start = (x, y)
            drawing = True
        else:
            drawing = False
            rect_end = (x, y)
            processed_image = param.copy()
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


def mode_pixelate(in_filename, out_filename):
    img = cv2.imread(in_filename)
    
    global rect_start, rect_end, drawing
    cv2.namedWindow("original_image")
    cv2.setMouseCallback("original_image", draw_rectangle, param=img)
    cv2.imshow('original_image', img)
    cv2.waitKey()
    
    cv2.destroyAllWindows() 


def main():
    args = cli_argument_parser()
    
    if(args.mode != 'resize' and args.width != -1):
        raise ValueError('Unknown argument \'width\'')
    if(args.mode != 'resize' and args.height != -1):
        raise ValueError('Unknown argument \'height\'')
    if(args.mode != 'vignette' and args.radius != -1):
        raise ValueError('Unknown argument \'radius\'')

    if args.mode == 'image':
        highgui_samples(args.image_path, args.out_image_path)
    elif args.mode == 'grayscale':
        mode_grayscale(args.image_path, args.out_image_path)
    elif args.mode == 'sepia':
        mode_sepia(args.image_path, args.out_image_path)
    elif args.mode == 'resize':
        if(args.width < 0 or args.height < 0):
            raise ValueError('Unspecified or invalid target width and height')
        mode_resize(args.image_path, args.out_image_path, args.width, args.height)
    elif args.mode == 'vignette':
        if(args.radius < 0):
            raise ValueError('Unspecified or invalid radius')
        mode_vignette(args.image_path, args.out_image_path, args.radius)
    elif args.mode == 'pixelate':
        mode_pixelate(args.image_path, args.out_image_path)
    else:
        raise ValueError('Unsupported mode')


if __name__ == '__main__':
    sys.exit(main() or 0)
