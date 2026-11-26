from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    histogram: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str = "Bag-of-Visual-Words histogram",
):
    """Plot a normalized BoW histogram and optionally save/show it."""
    values = np.asarray(histogram, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Histogram is empty")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(values.size), values, width=0.9)
    ax.set_title(title)
    ax.set_xlabel("Visual word index")
    ax.set_ylabel("Normalized frequency")
    ax.set_xlim(-0.5, values.size - 0.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def visualize_image_with_histogram(
    image: np.ndarray,
    histogram: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str = "Image and its BoW representation",
):
    """Display an image next to its normalized BoW histogram."""
    import cv2

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    values = np.asarray(histogram, dtype=np.float32).reshape(-1)
    fig, (image_ax, hist_ax) = plt.subplots(1, 2, figsize=(14, 5))
    image_ax.imshow(rgb)
    image_ax.set_title("Input image")
    image_ax.axis("off")

    hist_ax.bar(np.arange(values.size), values, width=0.9)
    hist_ax.set_title("Normalized BoW histogram")
    hist_ax.set_xlabel("Visual word index")
    hist_ax.set_ylabel("Frequency")
    hist_ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig
