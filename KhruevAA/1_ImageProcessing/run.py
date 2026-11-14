import argparse
import cv2

from filters import ImageFilter
from pathlib import Path


def read_image(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Error: File '{file_path}' does not exist.")
    
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Error: Unable to read file '{file_path}'. Format may not be supported.")
    return image


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Utility for applying image filters based on OpenCV and NumPy.")
    
    parser.add_argument('-i', '--input', required=True, type=Path, help="Path to the input image")
    parser.add_argument('-o', '--output', required=True, help="Path to save the output image")
    parser.add_argument('-f', '--filter', required=True, 
                        choices=['resize', 'grayscale', 'antique', 'fade', 'infrared', 'matte', 'noise', 'neon'], 
                        help="Filter type")
    
    parser.add_argument('--width', type=int, default=800, help="Width (for resize)")
    parser.add_argument('--height', type=int, default=600, help="Height (for resize)")
    parser.add_argument('--factor', type=float, default=0.6, help="Fade factor (0.0 - 1.0) for fade filter")
    
    return parser.parse_args()


def main():
    args = cli_argument_parser()
    
    try:
        image = read_image(args.input)
        
        kwargs = {}
        if args.filter == 'resize':
            kwargs = {'width': args.width, 'height': args.height}
        elif args.filter == 'fade':
            kwargs = {'fade_factor': args.factor}
            
        img_filter = ImageFilter.get_filter(args.filter, **kwargs)
        result = img_filter.apply_filter(image)
        
        cv2.imwrite(args.output, result)
        print(f"Success! Image processed with the '{args.filter}' filter and saved to '{args.output}'.")
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()