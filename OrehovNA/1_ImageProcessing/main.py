import os
import sys
import argparse
import cv2
from filters import ImageFilter


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="OpenCV Image Processing Filter CLI Utility"
    )
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument(
        '--filter', '-f', required=True,
        choices=['resize', 'gray', 'antique', 'fade', 'film',
                 'matte', 'scratch', 'neon'],
    )

    # --- resize ---
    parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)

    # --- fade ---
    parser.add_argument('--factor', type=float, default=0.4)

    # --- matte ---
    parser.add_argument('--matte_radius', type=float, default=0.7)
    parser.add_argument('--matte_softness', type=float, default=0.2)

    # --- scratch ---
    parser.add_argument('--noise', type=float, default=0.04)
    parser.add_argument('--scratches', type=int, default=7)

    # --- neon: только то, что нужно пользователю ---
    parser.add_argument('--neon_blur', type=float, default=0.8,
                        help="Base blur factor (scales everything else)")
    parser.add_argument('--core_brightness', type=float, default=1.8,
                        help="Brightness of the neon line itself")
    parser.add_argument('--aura_brightness', type=float, default=1.2,
                        help="Brightness of the glow around the line")
    parser.add_argument('--aura_scale', type=float, default=8.0,
                        help="Aura width: sigma = neon_blur * aura_scale")
    parser.add_argument('--core_gamma', type=float, default=0.65,
                        help="Line thickness: <1 thicker, >1 thinner")

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
            kwargs = {'noise': args.noise, 'scratches': args.scratches}
        elif args.filter == 'matte':
            kwargs = {'matte_radius': args.matte_radius,
                      'matte_softness': args.matte_softness}
        elif args.filter == 'neon':
            kwargs = {
                'neon_blur': args.neon_blur,
                'core_brightness': args.core_brightness,
                'aura_brightness': args.aura_brightness,
                'aura_scale': args.aura_scale,
                'core_gamma': args.core_gamma,
            }

        img_filter = ImageFilter.get_filter(args.filter, **kwargs)
        processed_img = img_filter.apply_filter(img)

        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(args.output, processed_img)
        print(f"Successfully saved output image to: {args.output}")

    except Exception as e:
        print(f"Error occurred during image processing execution: {e}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()