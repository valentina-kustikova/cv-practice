import argparse
import sys
import cv2 as cv
import numpy as np


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'ImageToGrayscale\', \'changingImageResolution\', \'SepiaImage\', \'VignetteImage\', \'PixelatingRectImage\')',
                        type=str,
                        dest='mode',
                        default='image')
    parser.add_argument('-i', '--image',
                        help='Path to an image',
                        type=str,
                        dest='image_path')
    parser.add_argument('-o', '--output',
                        help='Output image name',
                        type=str,
                        dest='output_image',
                        default='output.jpg')
    parser.add_argument('-rw', '--resolutionw',
                        help='change image resolution width',
                        type=int,
                        dest='width')
    parser.add_argument('-rh', '--resolutionh',
                        help='change image resolution height',
                        type=int,
                        dest='height')
    parser.add_argument('-rd', '--radius',
                        help='circle radius for vignette',
                        type=int,
                        dest='radius')
    parser.add_argument('-p', '--pixelate_size',
                        help='pixelate size',
                        type=int,
                        dest='pixelate_size')
    
    
    
    args = parser.parse_args()
    return args

def ImageToGrayscale(image):
               
    grayImage = np.zeros(image.shape)
    
    grayImage[:, :, 0] = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    grayImage[:, :, 1] = grayImage[:, :, 0]
    grayImage[:, :, 2] = grayImage[:, :, 0]

    grayImage = grayImage.astype(np.uint8)


    return grayImage
    
    
def changingImageResolution(image, width, height):
    

    x = np.linspace(0, image.shape[1] - 1, width)
    y = np.linspace(0, image.shape[0] - 1, height)
    x, y = np.meshgrid(x, y)

    x_coords = np.round(x).astype(int)
    y_coords = np.round(y).astype(int)

    result = image[y_coords, x_coords]

    return result
    
    
   

def SepiaImage(image):
       
   
    sepiaImage = np.zeros(image.shape)

    
    sepiaImage[:, :, 2] = 0.393 * image[:, :, 2] + 0.769 * image[:, :, 1] + 0.189 * image[:, :, 0]
    sepiaImage[:, :, 1] = 0.349 * image[:, :, 2] + 0.686 * image[:, :, 1] + 0.168 * image[:, :, 0]
    sepiaImage[:, :, 0] = 0.272 * image[:, :, 2] + 0.534 * image[:, :, 1] + 0.131 * image[:, :, 0]
   
    sepiaImage = np.clip(sepiaImage, 0, 255).astype(np.uint8)
    
    return sepiaImage
    
def VignetteImage(image, radius):

   
    height = image.shape[0]
    width  = image.shape[1]

    center_x, center_y = width // 2, height // 2

    mask = np.zeros((height, width), dtype=np.float32)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    
    mask = np.ones(distance.shape, dtype=np.float32)

    outside_circle = distance > radius
    distance_to_border = distance[outside_circle] - radius

    # масштаб затухания
    strength = 200
    mask[outside_circle] = np.exp(-(distance_to_border ** 2) / (strength * (10 ** 2)))

    mask = mask / np.max(mask)

    vignette = np.copy(image)
  
    vignette[:, :, 0] = vignette[:, :, 0] * mask
    vignette[:, :, 1] = vignette[:, :, 1] * mask
    vignette[:, :, 2] = vignette[:, :, 2] * mask

    return vignette


def pixelate_region(image, x1, y1, x2, y2, pixel_size):
    

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
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

    # Заменяем пикселизированную область на исходном изображении
    image[y1:y2, x1:x2] = roi


    
def draw_rectangle(event, x, y, flags, param):
    global rect, drawing, img, pixel_image, pixelate_size

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        rect = (x, y, x, y)
    
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img[:] = pixel_image.copy()  
            cv.rectangle(img, (rect[0], rect[1]), (x, y), (0, 255, 0), 2)
            rect = (rect[0], rect[1], x, y)
    
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        img[:] = pixel_image.copy()  
        pixelate_region(img, rect[0], rect[1], x, y, pixelate_size)
        pixel_image[:] = img.copy()  # Обновляем копию
        
def PixelatingRectImage(image, size):
    global img, pixel_image, rect, drawing, pixelate_size

    pixelate_size = size
    rect = (0, 0, 1, 1)
    drawing = False
    
   
    pixel_image = image.copy()
    img = image.copy()

    
    cv.namedWindow('Image')
    cv.setMouseCallback('Image', draw_rectangle)

    while True:
        cv.imshow('Image', img)
        if cv.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
            break

    
    
    
def main():
    
    args = cli_argument_parser()
    
    image_path = args.image_path
    if image_path is None:
        raise ValueError('Empty path to the image')

    image = cv.imread(image_path)
    
    
    if args.mode == 'ImageToGrayscale':
        result = ImageToGrayscale(image)
    elif args.mode == 'changingImageResolution':
        result = changingImageResolution(image, args.width, args.height)
    elif args.mode == 'SepiaImage':
        result = SepiaImage(image)
    elif args.mode == 'VignetteImage':
        result = VignetteImage(image, args.radius)
    elif args.mode == 'PixelatingRectImage':
        PixelatingRectImage(image, args.pixelate_size)
    else:
        raise 'Unsupported \'mode\' value'
        
    cv.imshow('Original image', image)
    cv.imshow('Output image', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
