import cv2
from cv2.typing import MatLike
import numpy as np

from labTypings import RGB, Rectangle


def resize(image: MatLike, width: int, height: int) -> MatLike:
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Parameters witdth and height must be positive: {width}, {height}"
        )

    oldHeight, oldWidth = map(int, image.shape[:2])  # pyright: ignore[reportAny]
    scaleX = float(oldWidth) / width
    scaleY = float(oldHeight) / height

    xGrid, yGrid = np.meshgrid(np.arange(width), np.arange(height))

    srcX = ((xGrid + 0.5) * scaleX - 0.5).astype(float)
    srcY = ((yGrid + 0.5) * scaleY - 0.5).astype(float)

    intX = srcX.astype(int)
    intY = srcY.astype(int)
    dx = srcX - intX
    dy = srcY - intY

    def clamp(val: np.typing.ArrayLike, maxVal: int):
        return np.clip(val, 0, maxVal - 1)

    p1 = image[clamp(intY, oldHeight), clamp(intX, oldWidth)]
    p2 = image[clamp(intY, oldHeight), clamp(intX + 1, oldWidth)]
    p3 = image[clamp(intY + 1, oldHeight), clamp(intX, oldWidth)]
    p4 = image[clamp(intY + 1, oldHeight), clamp(intX + 1, oldWidth)]

    interpolatedImg = (
        p1 * (1 - dx)[:, :, np.newaxis] * (1 - dy)[:, :, np.newaxis]
        + p2 * dx[:, :, np.newaxis] * (1 - dy)[:, :, np.newaxis]
        + p3 * (1 - dx)[:, :, np.newaxis] * dy[:, :, np.newaxis]
        + p4 * dx[:, :, np.newaxis] * dy[:, :, np.newaxis]
    ).astype(np.uint8)

    return interpolatedImg


def sepia(image: MatLike) -> MatLike:
    sepiaMatrix = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]],
        dtype=np.float32,
    )

    imageCopy = image.copy()

    sepiaImage = np.dot(imageCopy.astype(np.float32), sepiaMatrix.T)

    sepiaImage = np.clip(sepiaImage, 0, 255)

    return sepiaImage.astype(np.uint8)


def addVignette(image: MatLike, radius: float, opacity: float = 0.5) -> MatLike:
    height, width = map(int, image.shape[:2])  # pyright: ignore[reportAny]
    xGrid, yGrid = np.meshgrid(np.arange(width), np.arange(height))

    centerX = width // 2
    centerY = height // 2

    radX = int(radius * width / 2)
    radY = int(radius * height / 2)

    distanceFromCenter = (xGrid - centerX) ** 2 / radX**2 + (
        yGrid - centerY
    ) ** 2 / radY**2

    mask = np.exp(-distanceFromCenter / (2 * opacity))[:, :, np.newaxis].astype(float)

    return (image.copy().astype(float) * mask).astype(np.uint8)


def pixelate(
    image: MatLike,
    rect: Rectangle,
    blockSize: int,
) -> MatLike:
    x, y, w, h = rect

    pixelateImage = image.copy()

    region = pixelateImage[y : y + h, x : x + w].astype(np.uint8)

    for i in range(0, h, blockSize):
        for j in range(0, w, blockSize):
            block = region[i : i + blockSize, j : j + blockSize]

            blockColor = np.mean(block, axis=(0, 1)).astype(np.uint8)

            region[i : i + blockSize, j : j + blockSize] = blockColor

    pixelateImage[y : y + h, x : x + w] = region

    return pixelateImage


def addRectFrame(image: MatLike, borderColor: RGB, borderWidth: int) -> MatLike:
    height, width = map(int, image.shape[:2])  # pyright: ignore[reportAny]

    bgrBorderColor = borderColor[::-1]

    borderedImage = image.copy()

    topBottom = np.full((borderWidth, width, 3), bgrBorderColor, dtype=np.uint8)
    borderedImage[:borderWidth, :] = topBottom
    borderedImage[-borderWidth:, :] = topBottom

    leftRight = np.full((height, borderWidth, 3), bgrBorderColor, dtype=np.uint8)
    borderedImage[:, :borderWidth] = leftRight
    borderedImage[:, -borderWidth:] = leftRight

    return borderedImage


def addFigureFrame(image: MatLike, frameIndex: int, epsilon: float = 10) -> MatLike:
    height, width = map(int, image.shape[:2])  # pyright: ignore[reportAny]

    frame = cv2.imread(f"input/frame{frameIndex}.png")
    if frame is None:
        raise ValueError(f"Incorrect frameIndex: {frameIndex}")

    resizedFrame = resize(frame, width, height).astype(np.float32)

    whiteMask = np.logical_and.reduce(
        [
            (255.0 - resizedFrame[:, :, 0]) <= epsilon,
            (255.0 - resizedFrame[:, :, 1]) <= epsilon,
            (255.0 - resizedFrame[:, :, 2]) <= epsilon,
        ],
        axis=0,
    )

    combinedImage = np.where((~whiteMask)[:, :, np.newaxis], resizedFrame, image)

    return np.clip(combinedImage.astype(np.uint8), 0, 255)


def addGlare(image: MatLike, opacity: float = 0.5) -> MatLike:
    glare = cv2.imread("input/glare.jpg")
    if glare is None:
        raise ValueError("Have no glare.jpg")

    height, width = map(int, image.shape[:2])  # pyright: ignore[reportAny]
    glare = resize(glare, width, height).astype(np.float32)

    glareMask = glare / 255.0

    glareMask *= opacity

    combination = 255.0 - (255.0 - image.copy().astype(np.float32)) * (1.0 - glareMask)

    return combination.astype(np.uint8)


def addWatercolor(image: MatLike, opacity: float = 0.3) -> MatLike:
    watercolor = cv2.imread("input/watercolor.jpg")
    if watercolor is None:
        raise ValueError("Have no watercolor.jpg")
    if watercolor.shape[:2] != image.shape[:2]:
        watercolor = resize(watercolor, image.shape[1], image.shape[0])

    mask = 1.0 - watercolor / 255.0

    mask *= opacity

    combination = 255.0 - (255.0 - image.copy().astype(np.float32)) * (1.0 - mask)

    return combination.astype(np.uint8)
