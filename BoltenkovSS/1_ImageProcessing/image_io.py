import os

import cv2 as cv


def read_image(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at path: {filepath}")
    img = cv.imread(filepath)
    if img is None:
        raise ValueError(
            "Failed to decode the image. Ensure the file is a valid image format."
        )
    return img


def save_image(filepath, img):
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not cv.imwrite(filepath, img):
        raise IOError(f"Failed to save the image to: {filepath}")
