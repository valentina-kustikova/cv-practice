import numpy as np
import cv2 
import argparse

WIDTH = 1200
HEIGHT = 800

def cli_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', '-i',
                        help='Image path',
                        type=str,
                        dest='image_path',
                        required=False,
                        default='Miku.png')
    
    parser.add_argument('--filter', '-f',
                        help='filter type (gray, resize, sepia, vignette, pixelate)',
                        type=str,
                        dest='filter_type',
                        required=False,
                        default='gray', choices=['gray', 'resize', 'sepia', 'vignette', 'pixelate'])

    parser.add_argument('-p', '--param',
                        help='Parametres',
                        type=str,
                        dest='param')
    
    
    return parser.parse_args()

def rgb_2_gray(img):
    """
    Переводит из цветового режима RGB/BGR в Grayscale
    Args:
        img = редактируемое изображение
    retval:
        output = выходное изображение
    """
    r, g, b = cv2.split(img)
    output = (0.2126 * r + 0.587 * g + 0.144 * b) / 255
    return output

# при попытке уменьшенное изображение увеличить возникают артефакты
# Используется билинейная интерполяция
def re_size(img, size):
    """
    Изменениие размеров входного изображения
    с использование билинейной интерполяции
    Args:
        img = редактируемое изображение
        new_w = ширина выходного изображения
        new_h = высота выходного изображения
    retval:
        output = выходное изображение
    """
    new_w, new_h = size
    h, w = img.shape[:2]
    output = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    x_rat = w / new_w
    y_rat = h / new_h
    
    # целочисленные координаты
    x_ind = (np.arange(new_w) * x_rat).astype(np.float32)
    y_ind = (np.arange(new_h) * y_rat).astype(np.float32)

    x1 = np.clip(np.floor(x_ind).astype(np.int32), 0, w - 2)
    y1 = np.clip(np.floor(y_ind).astype(np.int32), 0, h - 2)
    # разница между исходными коорд. и целочисл. индексами
    # насколько далеко новое значение находится между ближайшими пикселями
    x_diff = x_ind - x1
    y_diff = y_ind - y1
    
    x2 = x1 + 1
    y2 = y1 + 1
    
    for i in range(3): #
        # извлекаем значения четырёх ближайших пикселей
        # используется индексация массивом
        top_left = img[y1[:, None], x1, i]
        top_right = img[y1[:, None], x2, i]
        bottom_left = img[y2[:, None], x1, i]
        bottom_right = img[y2[:, None], x2, i]
        # интерполяция по горизонтали
        top = top_left * (1 - x_diff) + top_right * x_diff 
        bottom = bottom_left * (1 - x_diff) + bottom_right * x_diff
        # интерполяция по вертикали
        output[:,:,i] = top * (1 - y_diff[:, None]) + bottom * y_diff[:, None]
    return output
    
def transform(img, kernel):
    b, g, r = cv2.split(img) # поделили на каналы

    # перемножаем вектор-канал с коэффициентами
    newB = np.clip(b * kernel[0][0] + g * kernel[0][1] + r * kernel[0][2], 0, 255)
    newG = np.clip(b * kernel[1][0] + g * kernel[1][1] + r * kernel[1][2], 0, 255)
    newR = np.clip(b * kernel[2][0] + g * kernel[2][1] + r * kernel[2][2], 0, 255)
    #присваиваем новые каналы
    img[:,:,0], img[:,:,1], img[:, :, 2] = newB, newG, newR
    return img
def effect_sepia(img):
    """
    Накладывает фотоэффект Сепии на входное изображение
    Args:
        img = редактируемое изображение
    retval:
        output = выходное изображение
    """
    # 1 method
    kernel = np.array([[0.131, 0.534, 0.272],[0.168, 0.686, 0.349],[0.189, 0.769, 0.393]]) #сетка коэффициентов для фотоэффекта
    output = transform(np.copy(img), kernel)
    return output
    
def effect_vignette(img, data):
    """
    Накладывает фотоэффект Виньетки на входное изображение
    Args:
        img = редактируемое изображение
        rad (pixels) = радиус внутри которого не будет применено затемнение
        strength (float) = сила затемнения
    retval:
        output = выходное изображение
    """
    rad, strength = data
    rows, cols = img.shape[:2]
    if (rad > cols or rad > rows or rad < 0): rad = 0.3 * min(cols, rows) # если радиус содержит недопустимые значения, то используется значение равное 30% от мин. стороны
    cent_x, cent_y = cols // 2, rows // 2
    y, x = np.ogrid[:rows, :cols]                           # возвращает разделённый 2 массива координат пикселей по высоте и ширине
    dist = np.sqrt((x - cent_x) ** 2 + (y - cent_y) ** 2)   # рассчитываем расстояние от центра до каждой точки

    vign = np.clip((dist - rad) / (max(cent_x, cent_y) - rad), 0, 1)    # если значение больше радиуса, обрезаем до 0. Иначе до 1
    vign = 1 - vign                                                     # Инверсия, чтобы затемнение было за радиусом
    vign = np.power(vign, strength)                                     # Сила затемнения
    
    vign = np.expand_dims(vign, axis=-1)                                # расширяем матицу для работы с цветным изображенем

    output = (img * vign).astype(np.uint8)
    #cv2.circle(output, (cent_x, cent_y), int(rad), (255, 255, 255), 2) # выделяется окружность вне которой идёт затемнение
    return output


def pixelate(img, block_size, coords):
    """
    Пикселизация выделенной области входного изображения.
    Изображение разбвается на блок размера block_size
    Все пиксели в каждом блоке заменяются средним значением цвета блока
    Args:
        img = редактируемое изображение
        block_size (pixels) = размер блока, на которых будет разбивать выделенную область
        coords = (x1, y1, x2, y2) - координаты выделенной области
    retval:
        output = выходное изображение
    """
    output = img.copy()
    crop_img = output[int(coords[1]):int(coords[1]+coords[3]),  int(coords[0]):int(coords[0]+coords[2])] 
    h, w = crop_img.shape[:2]
    
    #cv2.GaussianBlur(crop_img, (7,7), 0)

    # Перемещение пикселей по блокам
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = crop_img[i:min(i + block_size, h), j:min(j + block_size, w)]    # Определяем границы блока
            mean_color = np.mean(block, axis=(0, 1)).astype(np.uint8)               # Находим средний цвет
            crop_img[i:min(i + block_size, h), j:min(j + block_size, w)] = mean_color   # Заполняем блок средним цветом

    output[int(coords[1]):int(coords[1]+coords[3]),  int(coords[0]):int(coords[0]+coords[2])] = crop_img
    return output

def Parametr(args):
    str = args.param.split(",")
    param = [int(number) for number in str]
    paramlen = len(param)
    return param, paramlen
if __name__ == '__main__':

    args = cli_arguments()
    image_path = args.image_path
    input_image = cv2.imread(image_path)
    filter_type = args.filter_type
    
    match filter_type:
        case "gray":
            output_img = rgb_2_gray(input_image)
        case "resize":
            param, paramLen = Parametr(args)
            if paramLen!=2: raise ValueError('Add parameters')
            output_img = re_size(input_image, param)
        case "sepia":
            output_img = effect_sepia(input_image)
        case "vignette":
            param, paramLen = Parametr(args)
            if paramLen!=2: raise ValueError('Add parameters')
            output_img = effect_vignette(input_image, param)
        case "pixelate":
            param, paramLen = Parametr(args)
            if paramLen!=1: raise ValueError('Add parameters')
            coords = cv2.selectROI('select the area', input_image, showCrosshair=False)
            cv2.destroyWindow('select the area')
            output_img = pixelate(input_image, param[0], coords)
        case _:
            raise 'Unsuppored \'mode\' value'
    cv2.imshow('Original', input_image) 
    cv2.imshow('Output', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()