import argparse
from pathlib import Path

import cv2

from filters import FILTERS, apply_filter


def cli_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input/02_woman_face.png")
    parser.add_argument("--output", default="output")
    parser.add_argument("--filter", choices=FILTERS + ["all"], default="all")
    return parser


def read_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Image was not found or cannot be opened")

    return image


def save_image(path, image):
    cv2.imwrite(path, image)


def main():
    parser = cli_argument_parser()
    args = parser.parse_args()

    image = read_image(args.input)
    output_path = Path(args.output)

    if args.filter == "all":
        output_path.mkdir(exist_ok=True)

        for filter_name in FILTERS:
            result = apply_filter(image, filter_name)
            save_image(str(output_path / f"{filter_name}.png"), result)
    else:
        result = apply_filter(image, args.filter)
        save_image(str(output_path), result)


if __name__ == "__main__":
    main()
