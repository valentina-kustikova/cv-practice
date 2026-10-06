import argparse
import logging
import sys

import numpy as np
from PIL import Image

from filters import ImageFilter


def read_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def write_image(path: str, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def cli_argument_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image filters (PR #1)")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-f", "--filter", required=True,
                   choices=["resize", "grayscale", "antique", "fade",
                            "film", "matte", "scratches", "neon"])

    # resize
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    # общие
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--black-lift", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)

    # antique
    p.add_argument("--grain", type=float, default=0.25)

    # film
    p.add_argument("--mix", type=float, default=0.6)

    # matte
    p.add_argument("--softness", type=float, default=0.15)
    p.add_argument("--scale", type=float, default=0.9)

    # scratches
    p.add_argument("--n-scratches", type=int, default=8)
    p.add_argument("--noise-sigma", type=float, default=12.0)
    p.add_argument("--dust", type=float, default=0.004)

    # neon
    p.add_argument("--threshold", type=float, default=30.0)
    p.add_argument("--glow-radius", type=int, default=2)
    p.add_argument("--glow-strength", type=float, default=1.2)

    return p.parse_args()


def build_filter(name: str, a: argparse.Namespace) -> ImageFilter:
    if name == "resize":
        return ImageFilter.get_filter("resize", width=a.width, height=a.height)
    if name == "grayscale":
        return ImageFilter.get_filter("grayscale")
    if name == "antique":
        return ImageFilter.get_filter("antique", strength=a.strength,
                                      grain=a.grain, seed=a.seed)
    if name == "fade":
        return ImageFilter.get_filter("fade", strength=a.strength,
                                      black_lift=a.black_lift)
    if name == "film":
        return ImageFilter.get_filter("film", mix=a.mix,
                                      grain=a.grain, seed=a.seed)
    if name == "matte":
        return ImageFilter.get_filter("matte", softness=a.softness,
                                      scale=a.scale)
    if name == "scratches":
        return ImageFilter.get_filter("scratches",
                                      n_scratches=a.n_scratches,
                                      noise_sigma=a.noise_sigma,
                                      dust=a.dust,
                                      seed=a.seed)
    if name == "neon":
        return ImageFilter.get_filter("neon", threshold=a.threshold,
                                      glow_radius=a.glow_radius,
                                      glow_strength=a.glow_strength)
    raise ValueError("unknown filter: " + name)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = cli_argument_parser()

    try:
        image = read_image(args.input)
    except (OSError, ValueError) as e:
        logging.error("read failed: %s", e)
        return 1

    try:
        flt = build_filter(args.filter, args)
    except ValueError as e:
        logging.error("%s", e)
        return 1

    result = flt.apply_filter(image)

    try:
        write_image(args.output, result)
    except OSError as e:
        logging.error("write failed: %s", e)
        return 1

    logging.info("saved %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())