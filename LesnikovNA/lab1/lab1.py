import argparse
import sys
import cv2 as cv
import numpy as np

a, c, b, d = -1, -1, -1, -1
is_drawing = False 

def CliArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help="Specify the processing mode (e.g., 'image', 'gray', 'resolution', 'sepia', 'pixel', 'vignette')", type=str, dest='mode', default='image')
    parser.add_argument('-i', help="Provide the file path to the input image", type=str, dest='image_path')
    parser.add_argument('-o', help="Specify the name of the output image file", type=str, dest='output_image', default='output.jpg')
    parser.add_argument('-v', help="Scaling factor for resizing the image", type=float, dest='value')
    parser.add_argument('-p', help="Define the size of the pixelation blocks", type=int, dest='pixel_size')
    parser.add_argument('-r', help="Specify the radius for the vignette effect", type=int, dest='radius')
    parser.add_argument('-a', help="Set the starting x-coordinate for the region of interest [a, b]", type=int, dest='a')
    parser.add_argument('-b', help="Set the ending x-coordinate for the region of interest [a, b]", type=int, dest='b')
    parser.add_argument('-c', help="Set the starting y-coordinate for the region of interest [c, d]", type=int, dest='c')
    parser.add_argument('-d', help="Set the ending y-coordinate for the region of interest [c, d]", type=int, dest='d')
    args = parser.parse_args()
    return args

def ReadImage(image_path):
    if not image_path:
        raise ValueError("The provided image path is empty.")
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Failed to load the image from the specified path.")
    cv.imshow("Original Image", image)
    return image

def WriteImage(output_image, result_image):
    cv.imwrite(output_image, result_image)

def ShowImage(result_image):
    cv.imshow('Processed image', result_image)
    cv.waitKey(0)   
    cv.destroyAllWindows()

def Resize(image, scale_factor):
    if scale_factor <= 0:
        raise ValueError("Scale factor must be greater than zero")
    img_height, img_width, num_channels = image.shape
    resized_width = int(img_width * scale_factor)
    resized_height = int(img_height * scale_factor)
    width_indices = np.floor(np.arange(resized_width) / scale_factor).astype(int)
    height_indices = np.floor(np.arange(resized_height) / scale_factor).astype(int)
    resized_image = image[height_indices[:, None], width_indices]
    return resized_image

def Gray(image):
    img_height, img_width, num_channels = image.shape
    gray_image = np.zeros((img_height, img_width), np.uint8)
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]
    gray_coefficient = blue_channel * 0.114 + green_channel * 0.587 + red_channel * 0.299
    gray_image[:, :] = gray_coefficient
    return gray_image


def Sepia(image):
    img_height, img_width, num_channels = image.shape
    sepia_image = np.zeros((img_height, img_width, num_channels), np.uint8)
    sepia_image[:, :, 2] = np.clip(0.393 * image[:, :, 2] + 0.769 * image[:, :, 1] + 0.189 * image[:, :, 0], 0, 255)
    sepia_image[:, :, 1] = np.clip(0.349 * image[:, :, 2] + 0.686 * image[:, :, 1] + 0.168 * image[:, :, 0], 0, 255)
    sepia_image[:, :, 0] = np.clip(0.272 * image[:, :, 2] + 0.534 * image[:, :, 1] + 0.131 * image[:, :, 0], 0, 255)
    return sepia_image

def Vignette(image, radius):
    img_height, img_width = image.shape[:2]
    center_x = int(img_width / 2)
    center_y = int(img_height / 2)
    y_coords, x_coords = np.indices((img_height, img_width))
    distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    vignette_mask = 1 - np.minimum(1, distance / radius)
    vignette_image = (image * vignette_mask[:, :, np.newaxis]).astype(np.uint8)
    return vignette_image

def Pixel(image, pixel_size, a, b, c, d):
    num_channels = image.shape[2]
    pixelated_image = image.copy()
    for start_x in range(a, b, pixel_size):
        for start_y in range(c, d, pixel_size):
            end_x = min(start_x + pixel_size, b)
            end_y = min(start_y + pixel_size, d)
            pixel_region = image[start_y:end_y, start_x:end_x]
            average_color = np.zeros(num_channels, dtype=int)
            for channel_index in range(num_channels):
                average_color[channel_index] = int(np.mean(pixel_region[:, :, channel_index]))
            pixelated_image[start_y:end_y, start_x:end_x] = average_color
    return pixelated_image

def SetROICallback(event, x, y, flags, param):
    global a, c, b, d, is_drawing
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True
        a, c = x, y  
    elif event == cv.EVENT_MOUSEMOVE:
        if is_drawing:
            b, d = x, y  
    elif event == cv.EVENT_LBUTTONUP:
        is_drawing = False
        b, d = x, y  

def ShowROIonImage(image):
    global a, c, b, d
    cv.namedWindow('Image')
    cv.setMouseCallback('Image', SetROICallback)
    while True:
        img_copy = image.copy()
        if a != -1 and c != -1 and b != -1 and d != -1:
            cv.rectangle(img_copy, (a, c), (b, d), (0, 255, 0), 2) 
        cv.imshow('Image', img_copy)
        key = cv.waitKey(10)
        if key == ord('p'): 
            break
    cv.destroyWindow('Image')

def main():
    global a,b,c,d
    args = CliArgumentParser()
    image = ReadImage(args.image_path)
    if args.mode == 'gray':
        result_image = Gray(image)
    elif args.mode == 'pixel':
        if args.pixel_size is None:
            raise ValueError("The 'p' parameter is required for pixelization mode.")
        if (args.a != None and args.b != None and args.c != None and args.d != None):
            a = args.a
            b = args.b
            c = args.c 
            d = args.d 
        ShowROIonImage(image)
        result_image = Pixel(image, args.pixel_size, a, b, c, d)
    elif args.mode == 'resolution':
        if args.value is None:
            raise ValueError("Please provide the 'v' parameter for the resolution change mode.")
        result_image = Resize(image, args.value)
    elif args.mode == 'sepia':
        result_image = Sepia(image)
    elif args.mode == 'vignette':
        result_image = Vignette(image, args.radius)
    else:
        raise ValueError("The specified mode is not supported. Please check the 'mode' argument.")

    WriteImage(args.output_image, result_image)
    ShowImage(result_image)


if __name__ == '__main__':
    sys.exit(main() or 0)
