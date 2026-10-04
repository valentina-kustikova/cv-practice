import argparse
import cv2 as cv

def parse_args(args):
    """Отдельный разбор строки параметров типа width=124,height=124"""
    result = {}
    if args != "":
        for pair in args.split(","):
            key, value = pair.split("=")
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass               
            result[key.strip()] = value         
    return result

def cli_argument_parser():
    """Разбор аргументов из командной строки --filter --filter_args --image"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--filter", required=True, 
    choices=["rescale", "grayscale", "antique", "fade", "ir", "matte", "age", "neon"])
    
    parser.add_argument("--filter_args", default="")
    
    parser.add_argument("--image", required=True)
    
    args = parser.parse_args()
    return args
    
def read_image(image_name):
    """Чтение картинки по пути"""
    image = cv.imread(image_name, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image not found: {image_name}")
    return image
    
def save_image(filter_name, image_name, image):
    name = f"{filter_name}_{image_name}"
    cv.imwrite(name, image)