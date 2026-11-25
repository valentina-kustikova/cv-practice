
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _finish_figure(figure, save_path, show):
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(figure)


def plot_histogram(histogram, title="BoW-гистограмма", save_path=None, show=False):
    figure, axis = plt.subplots(figsize=(10, 4))
    word_indices = np.arange(len(histogram))
    axis.bar(word_indices, histogram, width=1.0, color="steelblue")
    axis.set_xlabel("Номер визуального слова")
    axis.set_ylabel("Частота")
    axis.set_title(title)
    axis.set_xlim(-0.5, len(histogram) - 0.5)
    _finish_figure(figure, save_path, show)


def visualize_image_with_histogram(image, histogram, predicted_class, save_path=None, show=False):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    figure, (image_axis, histogram_axis) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    image_axis.imshow(rgb)
    image_axis.set_title(f"Предсказанный класс: {predicted_class}")
    image_axis.axis("off")

    word_indices = np.arange(len(histogram))
    histogram_axis.bar(word_indices, histogram, width=1.0, color="steelblue")
    histogram_axis.set_xlabel("Номер визуального слова")
    histogram_axis.set_ylabel("Частота")
    histogram_axis.set_title("BoW-гистограмма")
    histogram_axis.set_xlim(-0.5, len(histogram) - 0.5)

    _finish_figure(figure, save_path, show)


def use_non_interactive_backend():
    """Переключает matplotlib на backend без окон (для сохранения в файл без показа)."""
    matplotlib.use("Agg")
