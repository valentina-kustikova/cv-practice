import numpy as np
import cv2 
import argparse


def resize(image,new_width,new_height):
    old_height, old_width = image.shape[:2] 
    # x_ratio = на сколько пикселей оригинала приходится на 1 пиксель нового изображения
    x_ratio = old_width / new_width  
    y_ratio = old_height / new_height 

    # Создаем сетку координат для нового изображения
    # np.arange(new_width) создает массив [0, 1, 2, ..., new_width-1]
    x_indices = (np.arange(new_width) * x_ratio).astype(np.int32)
    y_indices = (np.arange(new_height) * y_ratio).astype(np.int32)

    res_img = image[y_indices[:, None], x_indices]

    return res_img

def sepia(image, intensity=1.0):
   
    img_float = image.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],  # Новый синий: из оригинальных красных коэффициентов
        [0.168, 0.686, 0.349],  # Новый зеленый: из оригинальных зеленых коэффициентов
        [0.189, 0.769, 0.393]   # Новый красный: из оригинальных синих коэффициентов
    ])

    if intensity != 1.0:
        sepia_matrix = sepia_matrix * intensity + np.eye(3) * (1 - intensity)
    
    new_b = sepia_matrix[0, 0] * b + sepia_matrix[0, 1] * g + sepia_matrix[0, 2] * r
    new_g = sepia_matrix[1, 0] * b + sepia_matrix[1, 1] * g + sepia_matrix[1, 2] * r
    new_r = sepia_matrix[2, 0] * b + sepia_matrix[2, 1] * g + sepia_matrix[2, 2] * r
    
    sepia_image = np.stack([new_b, new_g, new_r], axis=2)
    res_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return res_image


def pixel_region(image, x, y, width, height, pixel_size=10):
   
    result = image.copy()
    img_height, img_width = image.shape[:2]
    
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    
    region = image[y:y+height, x:x+width]
    
    # Создаем пикселизированную область 
    pixelated_region = np.zeros_like(region)
    
    # Проходим по блокам размером pixel_size x pixel_size
    for block_y in range(0, height, pixel_size):
        for block_x in range(0, width, pixel_size):
            # Определяем границы текущего блока
            block_end_y = min(block_y + pixel_size, height)
            block_end_x = min(block_x + pixel_size, width)
            block_height = block_end_y - block_y
            block_width = block_end_x - block_x
            
            # Вычисляем средний цвет блока
            block = region[block_y:block_end_y, block_x:block_end_x]
            
            if block.size > 0:
                # Для цветного изображения
                if len(image.shape) == 3:
                    avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                # Для черно-белого изображения 
                    avg_color = np.mean(block).astype(np.uint8)
                
                # Заполняем весь блок средним цветом
                pixelated_region[block_y:block_end_y, block_x:block_end_x] = avg_color
    
   
    result[y:y+height, x:x+width] = pixelated_region
    
    return result


def vignette_gaussian(image, sigma=0.3, intensity=0.8):
    
    result = image.copy().astype(np.float32)
    height, width = image.shape[:2]
    

    center_x, center_y = width//2,height//2 
    
    y, x = np.ogrid[:height, :width]
    
    distance_x = (x - center_x) / (width * 0.5)
    distance_y = (y - center_y) / (height * 0.5)

    distance = np.sqrt(distance_x**2 + distance_y**2)

    # Гауссова маска: 1.0 в центре, экспоненциально убывает к краям
    vignette_mask = np.exp(-(distance**2) / (2 * sigma**2))
    
    # Нормализуем маску и применяем интенсивность
    vignette_mask = vignette_mask / vignette_mask.max()
    vignette_mask = 1.0 - (1.0 - vignette_mask) * intensity
    
    # Применяем к каждому каналу
    for channel in range(3):
        result[:, :, channel] *= vignette_mask
    
    res_img = np.clip(result, 0, 255).astype(np.uint8)

    return res_img


def add_border(image, border_width, color):
   
    height, width = image.shape[:2]
    
    new_height = height + 2 * border_width
    new_width = width + 2 * border_width
    
    if len(image.shape) == 3:
        # Цветное изображение (BGR)
        new_image = np.full((new_height, new_width, 3), color, dtype=np.uint8)
    else:
        # Черно-белое изображение
        new_image = np.full((new_height, new_width), color[0], dtype=np.uint8)
    
    new_image[border_width:border_width + height, 
              border_width:border_width + width] = image
    
    return new_image

def parse_border_color(color_str):
    try:
        b, g, r = map(int, color_str.split(','))
        return (b, g, r)
    except:
        print("Invalid color format. Using default white color.")
        return (255, 255, 255)  # Белый по умолчанию
    

def bliki(image, texture,k):
    texturen = cv2.imread(texture)
    h,w,_ = image.shape
    scaled_texture = resize(texturen,h,w)
    res = np.clip((image + k*scaled_texture),0,255).astype(np.uint8)
    return res

def watercolor_simple(img, texture_path, k=0.2):
    texture = cv2.imread(texture_path)
    h, w, _ = img.shape
    
    scaled_texture = resize(texture, h, w)

    inverted_texture = 255 - scaled_texture
    
    # Накладываем текстуру (вычитаем для эффекта бумаги)
    res = np.clip((img - k * inverted_texture), 0, 255).astype(np.uint8)
    
    return res

def shaped_border(image, frame_path):
    # Загружаем рамку с альфа-каналом
    frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Не удалось загрузить рамку: {frame_path}")
    
    img_height, img_width = image.shape[:2]
    
    frame_resized = resize(frame, img_width, img_height)
    
    result = image.copy()
    
    if frame_resized.shape[2] == 4:
        alpha = frame_resized[:, :, 3] 

        mask = alpha > 0
        
        for c in range(3):
            result[:, :, c] = np.where(mask, frame_resized[:, :, c], result[:, :, c])
    
    else:
        # Без альфа-канала - используем яркость для определения рамки
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Берем только НЕбелые пиксели (предполагаем, что рамка не белая)
        mask = gray_frame < 240  
        # Заменяем только пиксели рамки
        for c in range(3):
            result[:, :, c] = np.where(mask, frame_resized[:, :, c], result[:, :, c])
    
    return result

def cmd_args():#cmd/терминал команды,чтобы просить обработку конкретную

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', '-i',
                        help='Image path',
                        type=str,
                        dest='image_path',
                        required=False,
                        default="cat.jpg")

    parser.add_argument("--filters", "-f",
                        help="Filters type (grey, resize, sepia, vignette, pixelate)",
                        type=str,
                        dest="filters",
                        nargs ='+',
                        required=False,
                        default="pixelate",
                        choices=["resize", "sepia", "vignette", "pixelate","border","shaped_border","blick","watercolor"])
    
    parser.add_argument("--intensity", "-s", 
                        help="Sepia intensity (0.0 - 1.0)", 
                        type=float, 
                        dest="intensity", 
                        required=False, 
                        default=1.0)

    # для фильтра resize
    parser.add_argument("--width", "-w",
                        help="Width for resizing",
                        type=int,
                        dest="width",
                        required=False,
                        default=320)

    parser.add_argument("--height", "-ht",
                        help="Height for resizing",
                        type=int,
                        dest="height",
                        required=False,
                        default=320)
    #для пикселизации
    parser.add_argument("--width1", "-w1",
                        help="Width for pixelate",
                        type=int,
                        dest="width1",
                        required=False,
                        default=300)

    parser.add_argument("--height1", "-ht1",
                        help="Height for pixelate",
                        type=int,
                        dest="height1",
                        required=False,
                        default=300)
    parser.add_argument("--pixel_size", "-ps",
                        help="Pixel size for pixelate",
                        type=int,
                        dest="pixel_size",
                        required=False,
                        default=10)
    parser.add_argument("--xcoo", "-x",
                        help="XLeft for pixelate",
                        type=int,
                        dest="x",
                        required=False,
                        default=10)
    parser.add_argument("--ycoo", "-y",
                        help="YUpper for pixelate",
                        type=int,
                        dest="y",
                        required=False,
                        default=10)

    #для фильтра виньетки
    parser.add_argument("--sigma", "-sg",
                        help="Sigma (blur parameter) for vignette (0.0-1.0)",
                        type=float,
                        dest="sigma",
                        required=False,
                        default=0.3)
    parser.add_argument("--intensity1", "-s1",
                        help="Intensity for vignette (0.0-1.0)",
                        type=float,
                        dest="intensity1",
                        required=False,
                        default=0.8)
    
    #для рамки
    parser.add_argument("--border_width", "-bw",
                        help="Border width in pixels",
                        type=int,
                        dest="border_width",
                        required=False,
                        default=20)

    parser.add_argument("--border_color", "-bc",
                        help="Border color as BGR values (comma separated)",
                        type=str,
                        dest="border_color",
                        required=False,
                        default="255,255,255") 
    # для бликов
    parser.add_argument("--texture", "-t",
                        help="Texture for blick and for watercolor",
                        type=str,
                        dest="texture",
                        required=False,
                        default="blik.png")

    parser.add_argument("--coefficient", "-coef",
                        help="Coefficient for blick and watwrcolor",
                        type=float,
                        dest="coefficient",
                        required=False,
                        default=1.0) 
    #для фигурной рамки
    parser.add_argument("--shape","-sh",
                        help="Shaped border image",
                        type=str,
                        dest="shape",
                        required=False,
                        default="shaped_border.jpg")


    return parser.parse_args()


if __name__ == '__main__':
    args = cmd_args()

    image_path = args.image_path
    input_image = cv2.imread(image_path)

    if input_image is None:
        raise ValueError("Image not found or unable to load.")

    orig_name = "Original"
    cv2.imshow(orig_name, input_image)

    filter_type = args.filters
    for filter_name in args.filters: 
        if filter_name ==  "resize":
            output_image =resize(input_image, args.width, args.height)
            cv2.imshow("Resized Image", output_image)
        elif filter_name == "sepia":
           output_image = sepia(input_image,args.intensity)
           cv2.imshow("Sepia Image", output_image)
        elif filter_name == "vignette":
            output_image = vignette_gaussian(input_image,args.sigma,args.intensity1)
            cv2.imshow("Vignette Image", output_image)
        elif filter_name == "pixelate":
           output_image = pixel_region(input_image, args.x,args.y,args.width1,args.height1,args.pixel_size)
           cv2.imshow("Pixelate Image", output_image)
        elif filter_name == "border":
            border_color = parse_border_color(args.border_color)
            output_image = add_border(input_image, args.border_width, border_color)
            cv2.imshow("Bordered Image", output_image)
        elif filter_name == "blick":
            output_image = bliki(input_image, args.texture,args.coefficient)
            cv2.imshow("Blicked Image", output_image)
        elif filter_name == "watercolor":
            output_image = watercolor_simple(input_image,args.texture,args.coefficient)
            cv2.imshow("Watercolor effect",output_image)
        elif filter_name == "shaped_border":
            output_image = shaped_border(input_image,args.shape)
            cv2.imshow("Shape-bordered image",output_image)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()