from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as a BGR uint8 array; supports paths with non-ASCII characters."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except OSError as exc:
        raise OSError(f"Could not read image '{image_path}': {exc}") from exc

    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return image


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory of a file path and return the normalized path."""
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
