import sys

from cli import cli_argument_parser
from filters.base import ImageFilter
from image_io import read_image, save_image


def main():
    args = cli_argument_parser()

    try:
        img = read_image(args.input)

        filter_instance = ImageFilter.get_filter(args.filter)
        print(f"Applying '{args.filter}' filter...")
        params = {
            k: v
            for k, v in vars(args).items()
            if k not in ("input", "output", "filter")
        }
        result_img = filter_instance.apply_filter(img, **params)

        save_image(args.output, result_img)
        print(f"Success! Output saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
