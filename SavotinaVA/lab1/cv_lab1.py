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
    height, width, nchannels = image.shape
    gray = np.zeros((height, width, 1), dtype = np.uint8)
    
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            gr = 0.114 * b + 0.587 * g + 0.299 * r
            gray[i, j] = gr
            
            
    return gray


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
    height, width, nchannels = image.shape
    sepia = np.zeros((height, width, nchannels), dtype = np.uint8)
    
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            
            tr = 0.393*r + 0.769*g + 0.189*b
            tg = 0.349*r + 0.686*g + 0.168*b
            tb = 0.272*r + 0.534*g + 0.131*b
            
            if tr > 255:
                r = 255 
            else: 
                r = tr
            if tg > 255:
                g = 255 
            else: 
                g = tg
            if tb > 255:
                b = 255 
            else: 
                b = tb

            sepia[i, j] = b, g, r
                 
    return sepia


def Vignette(image):
    height, width, nchannels = image.shape
    center_x = int(height / 2)
    center_y = int(width / 2)
    
    final_img = image[:height, :width]
    for i in range(height):
        for j in range(width):
            val_x = 1 - np.abs(i - center_x) / center_x
            val_y = 1 - np.abs(j - center_y) / center_y
            final_img[i, j] = image[i, j] * val_x * val_y
    
    return final_img
    
    
def Pixelation(image, step):
    height, width = image.shape[:2]
    xstart = int(height / 5)
    xend = int(height / 2.5)
    ystart = int(width / 5)
    yend = int(width / 2.5)
    
    part_of_img = image[xstart:xend, ystart:yend]
    b, g, r = 0, 0, 0

    for i in range(xstart, xend, step):
        for j in range(ystart, yend, step):
            part_of_img = image[i:(i + step), j:(j + step)]
            b = np.mean(part_of_img[:, :, 0])
            g = np.mean(part_of_img[:, :, 1])
            r = np.mean(part_of_img[:, :, 2])
            
            for x in range(i, i + step):
                for y in range(j, j + step):
                    image[x, y] = b, g, r
    
    return image



def Gray_img(image_path, output_image):
    image = cv.imread(image_path)
    cv.imshow("Init image", image)
    
    gray_img = GrayShades(image)
    cv.imshow('Final image', gray_img)
    cv.waitKey(0)

    cv.imwrite(output_image, gray_img)  
    cv.destroyAllWindows()


def Resize_img(image_path, output_image, scale):
    image = cv.imread(image_path)
    cv.imshow("Init image", image)

    resize_img = Resize(image, scale)   
    cv.imshow('Final image', resize_img)
    cv.waitKey(0)

    cv.imwrite(output_image, resize_img)  
    cv.destroyAllWindows()


def Sepia_img(image_path, output_image):
    image = cv.imread(image_path)
    cv.imshow("Init image", image)
    
    sepia_img = Sepia(image)
    cv.imshow('Final image', sepia_img)
    cv.waitKey(0)

    cv.imwrite(output_image, sepia_img)
    cv.destroyAllWindows()
    
    
def Vignette_img(image_path, output_image):
    image = cv.imread(image_path)
    cv.imshow("Init image", image)
    
    vignette_img = Vignette(image)
    cv.imshow('Final image', vignette_img)
    cv.waitKey(0)

    cv.imwrite(output_image, vignette_img)
    cv.destroyAllWindows()
    
    
def Pixel_img(image_path, output_image, step):
    image = cv.imread(image_path)
    cv.imshow("Init image", image)
    
    pixel_img = Pixelation(image, step)
    cv.imshow('Final image', pixel_img)
    cv.waitKey(0)

    cv.imwrite(output_image, pixel_img)  
    cv.destroyAllWindows()
    


def main():
    args = arg_parser()
    
    if args.processing == "gray":
        Gray_img(args.image_path, args.output_image)
    elif args.processing == "resize":
        Resize_img(args.image_path, args.output_image, args.scale)
    elif args.processing == "sepia":
        Sepia_img(args.image_path, args.output_image)
    elif args.processing == "vignette":
        Vignette_img(args.image_path, args.output_image)
    elif args.processing == "pixelation":
        Pixel_img(args.image_path, args.output_image, args.step)
    else:
        raise 'Unsupported \'processing\' value'


if __name__ == '__main__':
    sys.exit(main() or 0)