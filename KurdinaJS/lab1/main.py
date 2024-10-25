import argparse
import sys
import cv2 as cv
import numpy as np

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode',
                        help='Mode (\'grayImage\', \'resolImage\', \'sepiaImage\', \'vignetteImage\', \'pixelImage\')',
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
    parser.add_argument('-v', '--video',
                        help='Path to a video file',
                        type=str,
                        dest='video_path')
    parser.add_argument('-p', '--param',
                        help='Parametres',
                        type=str,
                        dest='param')


    args = parser.parse_args()
    return args


def parsParam(args):
    str = args.param.split(",")
    param = [float(number) for number in str]
    numParam = len(param)
    return param, numParam


def readImage(image_path):
    if image_path is None:
        raise ValueError('Empty path to the image')
    image = cv.imread(image_path)
    return image
def outputImage(text, new_text, image, new_image):
    cv.imshow(text, image)
    cv.imshow(new_text, new_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def grayImage(image):

    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    gray_image[:, :] = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    gray_image = gray_image.astype(np.uint8)

    return gray_image


def resolutionImage(image, new_width, new_height):


    x = np.arange(new_width) / (new_width - 1) * (image.shape[1] - 1)
    y = np.arange(new_height) / (new_height - 1) * (image.shape[0] - 1)
    x, y = np.meshgrid(x, y)

    x_coords = np.round(x).astype(int)
    y_coords = np.round(y).astype(int)

    resolution_image = image[y_coords, x_coords]

    return resolution_image

def sepiaImage(image):

    sepia_image = np.zeros_like(image, dtype=np.float32)

    sepia_image[:, :, 0] = 0.393 * image[:, :, 2] + 0.769 * image[:, :, 1] + 0.189 * image[:, :, 0]
    sepia_image[:, :, 1] = 0.349 * image[:, :, 2] + 0.686 * image[:, :, 1] + 0.168 * image[:, :, 0]
    sepia_image[:, :, 2] = 0.272 * image[:, :, 2] + 0.534 * image[:, :, 1] + 0.131 * image[:, :, 0]

    sepia_image = np.clip(sepia_image, 0, 255)
    sepia_image = sepia_image.astype(np.uint8)
    sepia_image = cv.cvtColor(sepia_image, cv.COLOR_BGR2RGB)

    return sepia_image


def vignetteImage(image, radius, intensity):

    height = image.shape[0]
    width = image.shape[1]
    center_x = width // 2
    center_y = height // 2

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    norm = dist_from_center / (radius * max(center_x, center_y))
    norm = np.clip(norm, 0, 1)

    vignette = image * (1 - intensity * norm[..., np.newaxis])
    vignette = vignette.astype(np.uint8)

    return vignette


x1, y1, x2, y2 = -1, -1, -1, -1


def mouse_callback(event, x, y, flags, param):
    global x1, y1, x2, y2

    if event == cv.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if flags == cv.EVENT_FLAG_LBUTTON:
            x2, y2 = x, y
            image_copy = param.copy()
            cv.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imshow("BaseImage", image_copy)

def pixelImage(image, block_size):

    cv.namedWindow("BaseImage")
    global x1, y1, x2, y2

    cv.setMouseCallback("BaseImage", mouse_callback, param=image)
    cv.imshow("BaseImage", image)
    cv.waitKey(0)

    pixel_image = image.copy()
    epsilon = pixel_image[y1:y2, x1:x2]

    blocks_x = (x2 - x1) // block_size
    blocks_y = (y2 - y1) // block_size
    for i in range(blocks_y):
        for j in range(blocks_x):
            block = epsilon[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

            avg_color = np.mean(block, axis=(0, 1))

            epsilon[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = avg_color.astype(np.uint8)

    pixel_image[y1:y2, x1:x2] = epsilon

    return pixel_image


def main():
    args = cli_argument_parser()

    image = readImage(args.image_path)

    if args.mode == 'grayImage':
        filtImage = grayImage(image)
    elif args.mode == 'resolImage':
        param, numParam = parsParam(args)
        if numParam != 2:
            raise ValueError('Add parameters')
        filtImage = resolutionImage(image, int(param[0]), int(param[1]))
    elif args.mode == 'sepiaImage':
        filtImage = sepiaImage(image)
    elif args.mode == 'vignetteImage':
        param, numParam = parsParam(args)
        if numParam != 2:
            raise ValueError('Add parameters')
        filtImage = vignetteImage(image, param[0], param[1])
    elif args.mode == 'pixelImage':
        param, numParam = parsParam(args)
        if numParam != 1:
            raise ValueError('Add parameters')
        filtImage = pixelImage(image, int(param[0]))
    else:
        raise 'Unsupported \'mode\' value'

    outputImage('BaseImage', 'FilteredImage', image, filtImage)


if __name__ == '__main__':
    sys.exit(main() or 0)
