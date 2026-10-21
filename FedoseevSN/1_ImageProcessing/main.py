import argparse
import os
import sys

import cv2

from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Image processing")
    
    parser.add_argument("-i", "--input", required=True, help="Input image")
    parser.add_argument("-o", "--output", required=True, help="Output image")
    parser.add_argument(
        "-f", "--filter", required=True,
        choices=["resize", "gray", "antique", "fade", "film", "matte", "scratches", "neon"],
        help="Filter name",
    )
    
    parser.add_argument("--width", type=int, help="Width")
    parser.add_argument("--height", type=int, help="Height")
    parser.add_argument("--intensity", type=float, default=0.85)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--grain", type=float, default=15.0)
    parser.add_argument("--tint", type=float, default=0.4)
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--softness", type=float, default=0.2)
    parser.add_argument("--n-scratches", dest="n_scratches", type=int, default=20)
    parser.add_argument("--noise-level", dest="noise_level", type=float, default=15.0)
    parser.add_argument("--threshold", type=float, default=40.0)
    parser.add_argument("--glow-intensity", dest="glow", type=float, default=1.5)

    return parser.parse_args()


def get_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    return img


def gather_kwargs(args):
    if args.filter == "resize":
        if not args.width or not args.height:
            raise ValueError("Specify --width and --height for resize")
        return {"width": args.width, "height": args.height}
        
    params = {
        "antique": {"intensity": args.intensity},
        "fade": {"strength": args.strength},
        "film": {"grain": args.grain, "tint": args.tint},
        "matte": {"radius": args.radius, "softness": args.softness},
        "scratches": {"n_scratches": args.n_scratches, "noise_level": args.noise_level},
        "neon": {"threshold": args.threshold, "glow": args.glow},
    }
    return params.get(args.filter, {})


def main():
    args = cli_argument_parser()
    
    try:
        img = get_image(args.input)
        kwargs = gather_kwargs(args)
        
        img_filter = ImageFilter.get_filter(args.filter, **kwargs)
        processed = img_filter.apply_filter(img)
        
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        if not cv2.imwrite(args.output, processed):
            raise IOError(f"Failed to write: {args.output}")
            
        print(f"Saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()