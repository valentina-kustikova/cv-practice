import argparse


FILTER_NAMES = [
    "resize",
    "grayscale",
    "antique",
    "fade",
    "infrared",
    "matte",
    "aged",
    "neon",
]


def cli_argument_parser(argv=None):
    parser = argparse.ArgumentParser(
        description="Image processing tool using basic matrix operations."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    parser.add_argument(
        "-o", "--output", default="output.jpg", help="Path to save the output image"
    )
    parser.add_argument(
        "-f",
        "--filter",
        required=True,
        choices=FILTER_NAMES,
        help="Filter to apply",
    )

    parser.add_argument(
        "--scale_x", type=float, default=0.5, help="[resize] Scale factor X (> 0)"
    )
    parser.add_argument(
        "--scale_y", type=float, default=0.5, help="[resize] Scale factor Y (> 0)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="[fade] Weight of the original color, 0..1 (less = more faded)",
    )
    parser.add_argument(
        "--fade_level",
        type=int,
        default=150,
        help="[fade] Gray level the image is blended with, 0..255",
    )
    parser.add_argument(
        "--ir_gain",
        type=float,
        default=1.3,
        help="[infrared] Gain of the 'infrared' (red) channel, > 0",
    )
    parser.add_argument(
        "--matte_radius",
        type=float,
        default=0.8,
        help="[matte] Normalized ellipse radius where whitening starts, > 0",
    )
    parser.add_argument(
        "--matte_softness",
        type=float,
        default=0.2,
        help="[matte] Width of the smooth transition to white (0 = hard edge)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="[aged] Fraction of pixels for each of salt and pepper noise, 0..0.5",
    )
    parser.add_argument(
        "--scratches", type=int, default=15, help="[aged] Number of scratches, >= 0"
    )
    parser.add_argument(
        "--neon_gain",
        type=float,
        default=2.0,
        help="[neon] Brightness gain of the detected edges, > 0",
    )
    return parser.parse_args(argv)
