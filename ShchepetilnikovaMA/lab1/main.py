import argparse
import sys
import logging
import cv2 as cv
from filters import Filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('image_processor.log')
    ]
)
logger = logging.getLogger(__name__)

def read_image(path):
    image = cv.imread(path)
    if image is None:
        logger.error(f"Не удалось загрузить изображение: {path}")
        sys.exit(1)
    return image

def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', 
                        '--image',
                        help='Path to input image',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                         '--output',
                        help='Path to output image',
                        type=str,
                        default='output.jpg')
    parser.add_argument('-f', 
                        '--func',
                        help='Function to apply',
                        choices=['resize', 'sepia', 'vinetka', 'pixelize', 'rect_frame', 'frame', 'bliki', 'watercolor'],
                        required=True)
    parser.add_argument('--k',
                        type=float,
                        help='Parameter k (intensities for different filters)')
    parser.add_argument('--radius',
                        type=float,
                        help='Radius for vinetka')
    parser.add_argument('--new_size',
                        nargs=2,
                        type=int,
                        help='Size of new image for resize')
    parser.add_argument('--scale',
                        type=float,
                        help='The zoom level for the resize filter')
    parser.add_argument('--thickness',
                        type=int,
                        help='Thickness for rect_frame/frame')
    parser.add_argument( '--frame_type',
                        type=str,
                        choices=['circle', 'diamond', 'rectangle'],
                        help='Type of decorative frame: circle, diamond, rectangle',
                        default='circle')
    parser.add_argument('--frame',
                        type=str, 
                        help='Path to frame image')
    parser.add_argument('--texture',
                        type=str, 
                        help='Path to texture image')
    parser.add_argument('--color',
                        nargs=3,
                        type=int,
                        help='Color of the rectangle frame')
    parser.add_argument('--cx',
                        type=int,
                        help='X X coordinate for lens flare center')
    parser.add_argument('--cy',
                        type=int,
                        help='Y coordinate for lens flare center')
    parser.add_argument('--pixel_x',
                        type=int,
                        help='X coordinate for pixelate area')
    parser.add_argument('--pixel_y',
                        type=int,
                        help='Y coordinate for pixelate area')
    parser.add_argument('--pixel_w',
                        type=int,
                        help='Width for pixelate area')
    parser.add_argument('--pixel_h',
                        type=int,
                        help='Height for pixelate area')
    return parser


def apply_filter_by_args(args):
    parameters = {}
    if args.k is not None:
        parameters['intensity'] = args.k
        parameters['block_size'] = int(args.k)
    if args.radius is not None:
        parameters['strength'] = args.radius
    if args.new_size:
        parameters['width'], parameters['height'] = args.new_size
    if args.scale is not None:
        parameters['scale'] = args.scale * 100
    if args.thickness is not None:
        parameters['border_width'] = args.thickness
    if hasattr(args, 'frame_type'):
        parameters['frame_type'] = args.frame_type
    if args.color:
        parameters['color'] = tuple(args.color)
    if args.cx is not None:
        parameters['cx'] = args.cx
    if args.cy is not None:
        parameters['cy'] = args.cy
    if args.pixel_x is not None:
        parameters['x'] = args.pixel_x
    if args.pixel_y is not None:
        parameters['y'] = args.pixel_y
    if args.pixel_w is not None:
        parameters['width'] = args.pixel_w
    if args.pixel_h is not None:
        parameters['height'] = args.pixel_h
    return parameters

def main():
    parser = cli_argument_parser()
    args = parser.parse_args()
    image = cv.imread(args.image)
    logger.info(f"Загрузка изображения из: {args.image}")
    image = read_image(args.image)
    try:
        filter_instance = Filter.create_filter(args.func)
        parameters = apply_filter_by_args(args)
        logger.info(f"Применение фильтра {args.func} с параметрами: {parameters}")
        filtered_image = filter_instance.apply(image, parameters)
        cv.imshow("Original", image)
        cv.imshow("Filtered", filtered_image)
        logger.info("Нажмите любую клавишу для закрытия окон.")
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(args.output, filtered_image)
    except Exception as e:
        logger.error(f"Ошибка при применении фильтра: {e}")
        sys.exit(1)
    return 0

if __name__ == '__main__':
    sys.exit(main())