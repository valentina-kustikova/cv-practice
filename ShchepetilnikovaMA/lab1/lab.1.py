import argparse   
import sys
import cv2 as cv
import numpy as np 

#фильтры
#Функция изменения разрешения изображения.
def resize_image(image, scale_percent = None, height = None, width = None ):
    org_height, org_width = image.shape[:2]
    if scale_percent is not None:
        n_height = int(org_height * scale_percent /100)
        n_width = int(org_width * scale_percent /100)
    elif height is not None and  width is not None:
        n_height = height
        n_width = width
    else: 
        return image.copy()
    
    if scale_percent is not None and scale_percent < 100:
        interpolation = cv.INTER_AREA
    elif scale_percent is not None and scale_percent > 100:
        interpolation = cv.INTER_CUBIC
    else:
        interpolation = cv.INTER_LINEAR
    return cv.resize(image, (n_width, n_height), interpolation = interpolation)

#Функция применения фотоэффекта сепии к изображению.
def apply_sepia(image, intensy = 1.0):
    sepia_matrix = np.array([
        [0.272, 0.534, 0.131],  
        [0.349, 0.686, 0.168], 
        [0.393, 0.769, 0.189]  
    ])
    # Смешиваем с единичной матрицей для регулировки интенсивности
    identity_matrix = np.eye(3)
    if intensy <= 1.0:
        adjusted_matrix = intensy * sepia_matrix + (1 - intensy) * identity_matrix
    else:
        adjusted_matrix = sepia_matrix * intensy
        # Нормализуем, чтобы сумма коэффициентов в строке не превышала 1
        row_sums = adjusted_matrix.sum(axis=1, keepdims=True)
        adjusted_matrix = adjusted_matrix / np.maximum(row_sums, 1.0)
    sepia_image = cv.transform(image, adjusted_matrix)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

#Функция применения фотоэффекта виньетки к изображению.
#затемнение по краям, центр - всегда середина изображения, степень затемнения фиксир
def apply_vignette(image,strength = 0.5 ):
    height, width = image.shape[:2]
    center_x = width//2
    center_y  = height//2
    #создаём сетку координат, для каждого пикселя - его расстояние до центра
    x,y = np.ogrid[:height, :width]
    #гауссова маска затемнения
    x_new = (x - center_x)**2 / (2*(width*strength)**2)
    y_new = (y - center_y)**2 / (2*(height*strength)**2)
    mask = np.exp(-(x_new + y_new))
    mask = mask / mask.max()
    result = image.astype(np.float32)
    result = result * mask[:,:,np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)

#Функция пикселизации заданной прямоугольной области изображения.
def pixelate_area (image, x,y, height, width, block_size = 10):
    result = image.copy()
    if width is None:
        width = image.shape[1] - x
    if height is None:
        height = image.shape[0] - y
    #границы
    x = max(0, x)
    y = max(0, y)
    x_end = min(x + width, image.shape[1])
    y_end = min(y + height, image.shape[0])
    if x_end <= x or y_end<= y:
        print("invalid pixelate area")
        return result
    actual_width = x_end - x
    actual_height = y_end - y
    #вырезаем область
    place = result[y:y_end, x:x_end]
    block_size = max(2, min(block_size, min(actual_width, actual_height) // 2))
    #small = resize_image(place, scale_percent=100/block_size)
    small_width = max(1, actual_width // block_size)
    small_height = max(1, actual_height // block_size)
    small_place = cv.resize(place, (small_width, small_height), interpolation=cv.INTER_LINEAR)
    #pixelated = resize_image(small, scale_percent= block_size * 100)
    pixelated = cv.resize(small_place, (actual_width, actual_height), interpolation = cv.INTER_NEAREST)
    result[y:y_end, x:x_end] = pixelated
    return result

#Функция наложения прямоугольной одноцветной рамки заданной ширины по краям изображения.
def rect_border(image, border_width, color):
    result = image.copy()
    height, width = image.shape[:2]
    border_width = min(border_width, height//2, width//2)
    #обводка
    result[:border_width, :] = color   #верх
    result[-border_width:, :] = color  #низ
    result[:, :border_width ] = color  #лево
    result[:, -border_width:] = color  #право
    return result

#Функция наложения фигурной одноцветной рамки по краям изображения. Тип фигурной рамки является параметром функции.
def frame(image, bordered_type, bordered_width, color):
    result = image.copy()
    height, width = image.shape[:2]
    if bordered_type == "circle":
        center = (width//2, height//2)
        rad = min(height, width)//2 - bordered_width//2
        cv.circle(result, center, rad, color, bordered_width)
    elif bordered_type == "diamond":
        points = np.array([[width//2, 0],        #верх
                          [width, height//2],    #право
                          [width//2, height],    #низ
                          [0, height//2]         #лево
                          ], dtype = np.int32)
        cv.polylines(result, [points], color=color, thickness=bordered_width, isClosed=True)
    else:
        result[:bordered_width, :] = color
        result[-bordered_width:, :] = color
        result[:, :bordered_width] = color
        result[:, -bordered_width:] = color
    return result      

#Функция наложения эффекта бликов объектива камеры.
def bliks(image, cx = None, cy = None, intesy = 0.5 ):
    height, width = image.shape[:2]
    if cx is None:
        cx = width//2
    if cy is None:
        cy = height//2
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    flare = np.exp(-2 * distance / max_dist) * intesy
    flare = np.dstack([flare, flare, flare])
    result = image.astype(np.float32) + flare * 255
    return np.clip(result, 0, 255).astype(np.uint8)

#Функция наложения текстуры акварельной бумаги.
def watercolor(image, intesy = 0.2):
    height, width = image.shape[:2]
    # Генерируем случайный шум
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    # Делаем его мягким через размытие
    texture = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):
        texture[:, :, i] = cv.GaussianBlur(noise[:, :, i], (25, 25), 0)
    # Накладываем с прозрачностью
    return cv.addWeighted(image, 1 - intesy, texture, intesy, 0)

def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--image',
                        help='Path to input image',
                        type=str,
                        required=True)
    parser.add_argument('-o', '--output', help='Path to output image', type=str, default='output.jpg')
    parser.add_argument('-f', '--func', help='Function to apply', choices=['resize', 'sepia', 'vinetka', 'pixelize', 'rect_frame', 'frame', 'bliki', 'watercolor'], required=True)
    parser.add_argument('--k', type=float, help='Parameter k')
    parser.add_argument('--radius', type=float, help='Radius for vinetka')
    parser.add_argument('--new_size', nargs=2, type=int, help='Size of new image for resize')
    parser.add_argument('--scale', type=float, help='Scale for resize')
    parser.add_argument('--thickness', type=int, help='Thickness for rect_frame/frame')
    parser.add_argument('--frame_type', type=str, choices=['circle', 'diamond', 'rectangle'], help='Type of decorative frame: circle, diamond, rectangle', default='circle')
    parser.add_argument('--frame', type=str, help='Path to frame image')
    parser.add_argument('--texture', type=str, help='Path to texture image')
    parser.add_argument('--color', nargs=3, type=int, help='Color of the rectangle frame')
    parser.add_argument('--cx', type=int, help='X coordinate for lens flare center')
    parser.add_argument('--cy', type=int, help='Y coordinate for lens flare center')
    parser.add_argument('--pixel_x', type=int, help='X coordinate for pixelate area')
    parser.add_argument('--pixel_y', type=int, help='Y coordinate for pixelate area')
    parser.add_argument('--pixel_w', type=int, help='Width for pixelate area')
    parser.add_argument('--pixel_h', type=int, help='Height for pixelate area')
    return parser.parse_args()

def apply_filter_by_args(image, args):
    if args.func == "resize":
        if args.new_size:
            width, height = args.new_size
            return resize_image(image, width=width, height=height)
        elif args.scale is not None:
            return resize_image(image, scale_percent=args.scale * 100)
        else:
            return resize_image(image, scale_percent=50)
            
    elif args.func == "sepia":
        intensy = args.k if args.k is not None else 1.0
        return apply_sepia(image, intensy=intensy)
        
    elif args.func == "vinetka":
        strength = args.radius if args.radius is not None else 0.5
        return apply_vignette(image, strength=strength)
        
    elif args.func == "pixelize":
        # Пикселизация всей области по умолчанию
        block_size = int(args.k) if args.k is not None else 10
        # если есть заданная область
        if all([args.pixel_x is not None, args.pixel_y is not None, args.pixel_w is not None, args.pixel_h is not None]):
            print(f"Pixelating area: x={args.pixel_x}, y={args.pixel_y}, "f"width={args.pixel_w}, height={args.pixel_h}, block_size={block_size}")
            return pixelate_area(image, x=args.pixel_x, y=args.pixel_y, width=args.pixel_w, height=args.pixel_h, block_size=block_size)
        else:
            #пикселизируем всё изображение
            print(f"Pixelating entire image with block_size={block_size}")
            height, width = image.shape[:2]
            return pixelate_area(image, x=0, y=0, width=width, height=height, block_size=block_size)
        
    elif args.func == "rect_frame":
        thickness = args.thickness if args.thickness is not None else 10
        color = tuple(args.color) if args.color else (255, 255, 0)
        return rect_border(image, border_width=thickness, color=color)
        
    elif args.func == "frame":
        thickness = args.thickness if args.thickness is not None else 10
        color = tuple(args.color) if args.color else (255, 255, 0)
        frame_type = args.frame_type if hasattr(args, 'frame_type') else "circle"
        print(f"Applying {frame_type} frame with thickness {thickness} and color {color}")
        return frame(image, bordered_type=frame_type, bordered_width=thickness, color=color)
        
    elif args.func == "bliki":
        intesy = args.k if args.k is not None else 0.5
        if args.cx is not None and args.cy is not None:
            print(f"Using custom flare center: ({args.cx}, {args.cy})")
            return bliks(image, cx=args.cx, cy=args.cy, intesy=intesy)
        else:
            #иначе используем центр изображения 
            height, width = image.shape[:2]
            cx, cy = width // 2, height // 2
            print(f"Using image center: ({cx}, {cy})")
        return bliks(image, intesy=intesy)
        
    elif args.func == "watercolor":
        intesy = args.k if args.k is not None else 0.2
        return watercolor(image, intesy=intesy)
        
    else:
        raise ValueError(f"Unknown function: {args.func}")
    
def main():
    args = cli_argument_parser()
    
    image = cv.imread(args.image)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение '{args.image}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        filtered_image = apply_filter_by_args(image, args)
    except Exception as e:
        print(f"Ошибка при применении фильтра: {e}", file=sys.stderr)
        sys.exit(1)
    
    cv.imshow("Original", image)
    cv.imshow("Filtered", filtered_image)
    print("Нажмите любую клавишу для закрытия окон...")
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    cv.imwrite(args.output, filtered_image)
    print(f"Результат сохранён в '{args.output}'")

if __name__ == '__main__':
    sys.exit(main() or 0)
