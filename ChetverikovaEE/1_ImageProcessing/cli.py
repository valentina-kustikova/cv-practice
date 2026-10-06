import argparse


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Image filter application')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--filter', required=True,
                        choices=['resize', 'grayscale', 'antique', 'fade',
                                 'film', 'matte', 'scratches', 'neon'],
                        help='Filter type')
    parser.add_argument('--width', type=int, help='Width for resize')
    parser.add_argument('--height', type=int, help='Height for resize')
    parser.add_argument('--num_scratches', type=int, default=10,
                        help='Number of scratches')
    parser.add_argument('--noise_amount', type=float, default=0.05,
                        help='Noise amount for scratches')
    parser.add_argument('--contrast', type=float, default=0.5,
                            help='Contrast amount for fade')
    parser.add_argument('--alpha', type=float, default=0.3,
                            help='Alpha amount for matte and fade')
    parser.add_argument('--grain', type=float, default=0.05,
                                help='Grain amount for film')
    parser.add_argument('--threshold', type=float, default=40,
                                    help='Grain amount for neon')
    parser.add_argument('--glow_ksize', type=int, default=7,
                                    help='Grain amount for neon')
    return parser.parse_args()
