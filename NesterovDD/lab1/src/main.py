import argparse
from functools import partial

import cv2

import filters
from labTypings import RGB, Rectangle


def getFilterFunc(args: argparse.Namespace, imgHeight: int, imgWidth: int):
    if args.filterType == "resize":
        height: int | None = args.height
        width: int | None = args.width

        if height is None:
            height = imgHeight
        if width is None:
            width = imgWidth
        return partial(filters.resize, width=width, height=height)

    if args.filterType == "sepia":
        return filters.sepia

    if args.filterType == "vignette":
        opacity: float | None = args.opacity

        if opacity is None:
            return partial(filters.addVignette, radius=args.radius)
        else:
            return partial(filters.addVignette, radius=args.radius, opacity=opacity)

    if args.filterType == "pixelate":
        rect = Rectangle(*args.region)
        return partial(filters.pixelate, rect=rect, blockSize=args.blockSize)

    if args.filterType == "rectFrame":
        borderColor = RGB(*args.color)
        return partial(
            filters.addRectFrame, borderColor=borderColor, borderWidth=args.width
        )

    if args.filterType == "figureFrame":
        whiteThreshold: float | None = args.whiteThreshold

        if whiteThreshold is None:
            return partial(
                filters.addFigureFrame,
                frameIndex=args.frameIndex,
            )
        else:
            return partial(
                filters.addFigureFrame,
                frameIndex=args.frameIndex,
                epsilon=whiteThreshold,
            )

    if args.filterType == "glare":
        opacity: float | None = args.opacity

        if opacity is None:
            return filters.addGlare
        else:
            return partial(filters.addGlare, opacity=opacity)

    if args.filterType == "watercolor":
        opacity: float | None = args.opacity

        if opacity is None:
            return filters.addWatercolor
        else:
            return partial(filters.addWatercolor, opacity=opacity)

    raise ValueError("Filter not found")


def main(args: argparse.Namespace):
    imagePath = args.image

    image = cv2.imread(imagePath)

    if image is None:
        raise ValueError(f"There is no image: {imagePath}")

    filterFunc = getFilterFunc(args, imgWidth=image.shape[1], imgHeight=image.shape[0])

    filteredImage = filterFunc(image)

    cv2.imshow("Original", image)
    cv2.imshow("Filtered", filteredImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def createCliParser() -> argparse.ArgumentParser:
    argParser = argparse.ArgumentParser(prog="cvFilter")
    _ = argParser.add_argument("--image", type=str, required=True, help="Path to image")

    subparsers = argParser.add_subparsers(
        dest="filterType", title="Filters", help="A filter applied to the image"
    )

    resizeParser = subparsers.add_parser("resize")
    _ = resizeParser.add_argument(
        "--height", type=int, help="If not specified, original height will be used"
    )
    _ = resizeParser.add_argument(
        "--width", type=int, help="If not specified, original height will be used"
    )

    _ = subparsers.add_parser("sepia")
    vignetteParser = subparsers.add_parser("vignette")
    _ = vignetteParser.add_argument(
        "--radius", type=float, required=True, help="Value from 0 to 1"
    )
    _ = vignetteParser.add_argument("--opacity", type=float)

    pixelateParser = subparsers.add_parser("pixelate")
    _ = pixelateParser.add_argument(
        "--region",
        type=int,
        help="Coordiantes of top left cornern, width and height",
        required=True,
        nargs=4,
    )
    _ = pixelateParser.add_argument("--blockSize", type=int, required=True)

    rectFrameParser = subparsers.add_parser("rectFrame")
    _ = rectFrameParser.add_argument(
        "--color", type=int, help="RGB", required=True, nargs=3
    )
    _ = rectFrameParser.add_argument("--width", type=int, required=True)

    figureFrameParser = subparsers.add_parser("figureFrame")
    _ = figureFrameParser.add_argument("--frameIndex", type=int, required=True)
    _ = figureFrameParser.add_argument("--whiteThreshold", type=float)

    glareParser = subparsers.add_parser("glare")
    _ = glareParser.add_argument("--opacity", type=float)

    watercolorParser = subparsers.add_parser("watercolor")
    _ = watercolorParser.add_argument("--opacity", type=float)

    return argParser


if __name__ == "__main__":
    argParser = createCliParser()
    main(argParser.parse_args())
