from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_histogram(histogram, output_path=None, title="Гистограмма BoW", show=False):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(histogram)), histogram, width=1.0)
    ax.set(title=title, xlabel="Визуальное слово", ylabel="Нормированная частота")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_image_with_histogram(
    image,
    histogram,
    output_path=None,
    title="Результат классификации",
    show=False,
):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": (2, 1)}
    )

    ax1.imshow(rgb)
    ax1.set_title(title)
    ax1.axis("off")
    ax2.bar(np.arange(len(histogram)), histogram, width=1.0)
    ax2.set_xlabel("Визуальное слово")
    ax2.set_ylabel("Нормированная частота")
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
