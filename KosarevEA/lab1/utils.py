import cv2
import argparse
import os

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image

# Функция для вызова событий мыши
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['drawing'] = True
        param['start_point'] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if param['drawing']:
            param['end_point'] = (x, y)
            image_copy = param['image'].copy()
            cv2.rectangle(image_copy, param['start_point'], param['end_point'], (0, 255, 0), 2)
            cv2.imshow('Select Region', image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        param['drawing'] = False
        param['end_point'] = (x, y)

# Функция для выбора области
def select_region(image):
    params = {'drawing': False, 'start_point': None, 'end_point': None, 'image': image}
    cv2.imshow('Select Region', image)
    cv2.setMouseCallback('Select Region', mouse_callback, param=params)
    cv2.waitKey(0)
    cv2.destroyWindow('Select Region')

    # Возвращаем координаты выбранной области
    if params['start_point'] and params['end_point']:
        return (params['start_point'][0], params['start_point'][1],
                params['end_point'][0] - params['start_point'][0], 
                params['end_point'][1] - params['start_point'][1])
    else:
        return None



def parse_arguments():
    parser = argparse.ArgumentParser(description='Apply various filters to an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--filter', type=str,
                        choices=['vignette', 'pixelate', 'grayscale', 'resize', 'sepia'], required=True,
                        help='Filter to apply: vignette, pixelate, grayscale, resize or sepia')
    

    parser.add_argument('--radius', type=float, default=1.5,
                    help='Radius of the vignette effect (default: 1.5)')
    parser.add_argument('--intensity', type=float, default=1.0,
                    help='Intensity of the vignette effect (default: 1.0)')
    
    parser.add_argument('--pixel_size', type=int, default=10,
                        help='Size of pixels for pixelation (default: 10)')
    
    parser.add_argument('--resize_width', type=int,
                        help='Width for resizing')
    
    parser.add_argument('--resize_height', type=int,
                        help='Height for resizing')
    
    return parser.parse_args()
