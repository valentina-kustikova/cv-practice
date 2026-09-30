import argparse
import cv2

def resize(src, new_width, new_height):
    image = cv2.imread(src) #src = path
    resized_image = cv2.resize(image, (new_width, new_height) , dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
    return resized_image

def save_img(img, output_path):
    cv2.imwrite(output_path, img)

parser = argparse.ArgumentParser()
parser.add_argument("-hi", "--hello", help="print hello world", action="store_true")
parser.add_argument("-i", "--input", help="input a path to the image")
parser.add_argument("-o", "--output", help="input a path to where the image is gonna be saved")
parser.add_argument("-r", "--resize", nargs=2, metavar=('width', 'height'), type=int, help="resizes the image to the desired size")
args = parser.parse_args()

if args.hello:
    print("hello world")
if args.input and args.output:
    width, height = args.resize
    resized_image = resize(args.input, width, height)
    save_img(resized_image, args.output)
