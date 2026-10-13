# src/main.py
import os
import sys
import traceback

# Чтобы можно было запускать как `python src/main.py`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli import cli_argument_parser
from image_io import read_image, save_image
from filters import ImageFilter


def build_filter_kwargs(args):
    name = args.filter.lower()
    if name == "resize":
        return {"width": args.width, "height": args.height}
    if name == "antique":
        return {"vignette": args.vignette}
    if name in ("fade", "fadecolor"):
        return {"alpha": args.alpha, "contrast": args.contrast, "haze": args.haze}
    if name in ("film", "ir"):
        return {"grain": args.grain, "seed": args.seed}
    if name in ("matte", "oval"):
        return {"a_ratio": args.a_ratio, "b_ratio": args.b_ratio, "feather": args.feather}
    if name in ("scratches", "old", "aged"):
        return {
            "n_scratches": args.n_scratches,
            "scratch_intensity": args.scratch_intensity,
            "noise_level": args.noise_level,
            "seed": args.seed,
        }
    if name in ("neon", "contour"):
        return {
            "darkness": args.darkness,
            "neon_color": tuple(args.neon_color),
            "intensity": args.intensity,
        }
    return {}


def main():
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"[Ошибка чтения] {e}", file=sys.stderr)
        return 1

    try:
        filt = ImageFilter.get_filter(args.filter, **build_filter_kwargs(args))
        result = filt.apply_filter(image)
    except Exception as e:
        print(f"[Ошибка применения фильтра] {e}", file=sys.stderr)
        traceback.print_exc()
        return 2

    try:
        save_image(args.output, result)
    except IOError as e:
        print(f"[Ошибка сохранения] {e}", file=sys.stderr)
        return 3

    print(f"OK: {args.input} --[{args.filter}]--> {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())