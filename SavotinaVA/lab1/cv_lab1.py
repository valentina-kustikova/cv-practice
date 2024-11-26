import numpy as np
import cv2 as cv
import argparse
import sys

def arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input',
                        help = 'Path to an image',
                        type = str,
                        dest = 'image_path')
    
    parser.add_argument('-p', '--processing',
                    help = 'Processing (\'gray\', \'resize\', \'sepia\', \'vignette\', \'pixelation\')',
                    type = str,
                    dest = 'processing')
    
    parser.add_argument('-sc', '--scale',
                        help = 'Scale for resize',
                        type = float,
                        dest = 'scale',
                        default = 0.5)
    
    parser.add_argument('-r', '--radius',
                        help = 'Radius for vignette',
                        type = float,
                        dest = 'radius',
                        default = 1)
    
    parser.add_argument('-st', '--step',
                        help = 'Step for pixelation',
                        type = int,
                        dest = 'step',
                        default = 20)
    
    parser.add_argument('-o', '--output',
                       help = 'Output an image',
                       type = str,
                       dest = 'output_image',
                       default = 'output.jpg')
    
    args = parser.parse_args()
    return args


def GrayShades(image):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    
    gr = 0.114 * b + 0.587 * g + 0.299 * r  
            
    image[:, :, 0] = gr
    image[:, :, 1] = gr
    image[:, :, 2] = gr
    
    return image


def Resize(image, scale):
    height, width, nchannels = image.shape
    
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_img = np.zeros((new_height, new_width, nchannels), dtype = np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            new_img[i, j] = image[int(i / scale), int(j / scale)]
            
    return new_img


def Sepia(image):
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    
    tr = 0.393*r + 0.769*g + 0.189*b
    tg = 0.349*r + 0.686*g + 0.168*b
    tb = 0.272*r + 0.534*g + 0.131*b
            
    image[:, :, 0] = tb
    image[:, :, 1] = tg
    image[:, :, 2] = tr
                 
    return image


def Vignette(image, rad):
    height, width = image.shape[:2]

    x = int(height / 2)
    y = int(width / 2)
    
    a = 1
    b = 1
    
    if height > width:
        a = b * height / width
    elif height < width:
        b = a * width / height

    x_idx, y_idx = np.indices((height, width))

    dist = np.sqrt(((x_idx - x) / a) ** 2 + ((y_idx - y) / b) ** 2)
    coef = 1 - np.minimum(1, dist / rad)

    final_img = image[:height, :width]
    final_img[:, :, 0] = image[:, :, 0] * coef
    final_img[:, :, 1] = image[:, :, 1] * coef
    final_img[:, :, 2] = image[:, :, 2] * coef

    return final_img.astype(np.uint8)
        
      
def Pixelation(image, step, start, end):
    region = image[start[1]:end[1], start[0]:end[0]]
    
    height, width = region.shape[:2]
    copy_region = region[:height, :width]
    final_image = region[:height, :width]

    for i in range(0, height - 1, step):
        for j in range(0, width - 1, step):
            if i + step < height:
                border_y = i + step
            else:
                border_y = height - 1
            if j + step < width:
                border_x = j + step
            else:
                border_x = width - 1
            copy_region = region[i:border_y, j:border_x]
            mean = [np.mean(copy_region[:, :, 0]), np.mean(copy_region[:, :, 1]), np.mean(copy_region[:, :, 2])]
            final_image[i:border_y, j:border_x] = mean
    
    image[start[1]:end[1], start[0]:end[0]] = final_image
    return image       
      
      
def OnMouseClick(event, x, y, flags, param):
    global start, end, selection_done, drawing
    img, step= param

    if (selection_done == False):
        if event == cv.EVENT_LBUTTONDOWN:
            start[0], start[1] = x, y
            drawing = True
            selection_done = False
            
        elif event == cv.EVENT_MOUSEMOVE and drawing:
            end[0], end[1] = x, y
            img_copy = img.copy()
            cv.rectangle(img_copy, (start[0], start[1]), (end[0], end[1]), (255, 255, 255), 1)
            cv.imshow('Init image', img_copy)
            
        elif event == cv.EVENT_LBUTTONUP:
            end[0], end[1] = x, y
            drawing = False
            selection_done = True
            cv.imshow('Init image', img)
      
    
def SetRectangle(img, step):
    global start, end, selection_done

    cv.setMouseCallback('Init image', OnMouseClick, [img, step])
    
    while not selection_done:
       cv.waitKey(1)


def ReadImage(image_path):
    image = cv.imread(image_path)
    cv.imshow("Init image", image)
    
    return image

def WriteImage(img, output_image = "output.jpg"):   
    cv.waitKey(0)
    cv.imwrite(output_image, img)
    cv.destroyAllWindows()


start = [0, 0]
end = [0, 0]
selection_done = False
drawing = False

def main():
    args = arg_parser()
    image = ReadImage(args.image_path)
    
    if args.processing == "gray":
        output_img = GrayShades(image)
    elif args.processing == "resize":
        output_img = Resize(image, args.scale)
    elif args.processing == "sepia":
        output_img = Sepia(image)
    elif args.processing == "vignette":
        output_img = Vignette(image, args.radius)
    elif args.processing == "pixelation":
        SetRectangle(image, args.step)
        output_img = Pixelation(image, args.step, start, end)
    else: 
        raise 'Unsupported \'processing\' value'

    cv.imshow('Final image', output_img)
    WriteImage(output_img, args.output_image)

if __name__ == '__main__':
    sys.exit(main() or 0)