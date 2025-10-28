import argparse
import sys
import cv2 as cv
import logging
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
        raise ValueError('Incorrect or empty path to the image')
    src_image = cv.imread(image)
    cv.imshow("original", src_image)
    cv.waitKey(0)

    return src_image


def select_filter(src_image, filter, width, height, roi,
                  frame_width, frame_color, frame_type,
                  amplitude, frequency, pattern_size):
    if filter == 'resize':
        if not width or not height:
            raise ValueError("To resize, you must specify --width and --height")
        result_image = Filters.resize_image(src_image, (width, height))
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'sepia':
        result_image = Filters.sepia(src_image)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'vignette':
        result_image = Filters.vignette(src_image)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'pixelation_roi':
        if not roi or len(roi) != 4:
            raise ValueError("For pixelation_roi you need to specify --roi x1 y1 x2 y2")
        x1, y1, x2, y2 = roi
        result_image = Filters.pixelation_of_roi(src_image, x1, y1, x2, y2)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'rectangular_frame':
        result_image = Filters.rectangular_frame(src_image, frame_width, frame_color)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'figured_frame':
        result_image = Filters.figured_frame(src_image, frame_type, frame_width,
                                     frame_color, amplitude, frequency, pattern_size)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'lens_flare':
        result_image = Filters.lens_flare(src_image)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    elif filter == 'paper_texture':
        result_image = Filters.paper_texture(src_image)
        cv.imshow("new image", result_image)
        cv.waitKey(0)
    else:
        raise ValueError ("Unsupported \'filter\' value")
    cv.destroyAllWindows()


def main():
    args = cli_argument_parser()
    try:
        src_image = image_read(args.image)
        select_filter(src_image, args.filter, args.width, args.height, args.roi,
                  args.frame_width, args.frame_color, args.frame_type,
                  args.amplitude, args.frequency, args.pattern_size)
    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    sys.exit(main() or 0)
