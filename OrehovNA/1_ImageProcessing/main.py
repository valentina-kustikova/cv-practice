import os
import sys
import argparse
import cv2
from filters import ImageFilter

def cli_argument_parser():
    parser = argparse.ArgumentParser(description="OpenCV Image Processing Filter CLI Utility")
    parser.add_argument('--input', '-i', required=True, help="Path to input image file")
    parser.add_argument('--output', '-o', required=True, help="Path to save output image file")
    parser.add_argument('--filter', '-f', required=True, choices=['resize', 'gray', 'antique', 'fade', 'film', 'matte', 'scratch', 'neon'], help="Type of filter to apply")
    
    parser.add_argument('--scale', type=float, default=None, help="Scale factor for resizing")
    parser.add_argument('--width', type=int, default=None, help="Target width for resizing")
    parser.add_argument('--height', type=int, default=None, help="Target height for resizing")
    parser.add_argument('--factor', type=float, default=0.4, help="Fade color factor (0.0 to 1.0)")
    parser.add_argument('--noise', type=float, default=0.05, help="Noise intensity level for scratch filter")
    parser.add_argument('--scratches', type=int, default=5, help="Number of scratches for scratch filter")
    
    return parser.parse_args()

def read_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file not found at path: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to decode or parse image file: {image_path}")
    return image

def main():
    args = cli_argument_parser()
    
    try:
        print(f"Reading image from: {args.input}")
        img = read_image(args.input)
        
        kwargs = {}
        if args.filter == 'resize':
            kwargs = {'scale': args.scale, 'width': args.width, 'height': args.height}
        elif args.filter == 'fade':
            kwargs = {'factor': args.factor}
        elif args.filter == 'scratch':
            kwargs = {'noise_level': args.noise, 'scratches': args.scratches}
            
        img_filter = ImageFilter.get_filter(args.filter, **kwargs)
        
        processed_img = img_filter.apply_filter(img)
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        cv2.imwrite(args.output, processed_img)
        print(f"Successfully saved output image to: {args.output}")
        
    except Exception as e:
        print(f"Error occurred during image processing execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
