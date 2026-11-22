
import sys
import os
import cv2

from cli import cli_argument_parser
from filter import ImageFilter

def read_image(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Image not found: {filename}")
    img = cv2.imread(filename)
    if img is None:
        raise ValueError(f"Failed to read image: {filename}")
    return img


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    try:
        ImageFilter.bind_args(args)
        filt = ImageFilter.get_filter(args.filter)
        result = filt.apply_filter(image)
        cv2.imwrite(args.output, result)
        print(f"Filter '{args.filter}' applied successfully. Saved to {args.output}")
    except Exception as e:
        print(f"Error applying filter: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()