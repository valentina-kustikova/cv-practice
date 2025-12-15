import argparse
import sys
import cv2 as cv
import logging
from pathlib import Path
from filters import Filters

def cli_argument_parser():
    parser = argparse.ArgumentParser(
        prog = 'Filters',
        description='Image processing with various filters'
    )
    parser.add_argument('-i', '--image',
                        required=True,
                        help='Path to an image',
                        type=str,
                        dest='image')
    parser.add_argument('--filter', '-f',
                        required=True,
                        choices=['resize', 'sepia', 'vignette', 'pixelation_roi',
                                 'rectangular_frame', 'figured_frame', 'lens_flare', 'paper_texture'],
                        help='Selecting a filter to apply',
                        type=str,
                        dest='filter')
    parser.add_argument('--width',
                        type=int,
                        help='New width')
    parser.add_argument('--height',
                        type=int,
                        help='New height')
    parser.add_argument('--roi',
                        nargs=4,
                        type=int,
                        help='ROI for pixelization: x1 y1 x2 y2')
    parser.add_argument('--frame_width',
                        type=int,
                        default=30,
                        help='Frame width')
    parser.add_argument('--frame_type',
                        choices=['wavy', 'zigzag', 'diagonal'],
                        help='Frame type')
    parser.add_argument('--frame_color',
                        nargs=3,
                        type=int,
                        default=[0, 0, 255],
                        metavar=('B', 'G', 'R'),
                        help='Frame color in B G R format')
    parser.add_argument('--amplitude',
                        type=int,
                        default=10,
                        help='Amplitude for the wavy frame')
    parser.add_argument('--frequency',
                        type=float,
                        default=0.1,
                        help='Frequency for the wavy frame')
    parser.add_argument('--pattern_size',
                        type=int,
                        default=20,
                        help='Pattern size for zigzag and diagonal frame')
    args = parser.parse_args()

    return args


def image_read(image):
    if image is None:
        raise ValueError('Empty path to the image')
    filepath = Path(image)
    if not filepath.exists():
        raise ValueError('Incorrect path to the image')
    src_image = cv.imread(image)
    cv.imshow("original", src_image)
    cv.waitKey(0)

    return src_image


def main():
    args = cli_argument_parser()
    try:
        src_image = image_read(args.image)
        Filters_instance = Filters()
        res_image = Filters_instance.apply_filter(src_image, args.filter, args.width, args.height, args.roi,
                  args.frame_width, args.frame_color, args.frame_type,
                  args.amplitude, args.frequency, args.pattern_size)
        cv.imshow("new image", res_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except Exception as e:
        logging.error(e)
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main() or 0)
