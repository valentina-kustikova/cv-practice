from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np

if sys.platform.startswith("linux") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir

logger = logging.getLogger(__name__)

BAR_COLOR = "#2f6db5"


def plot_histogram(histogram: np.ndarray, title: str = "BoW-гистограмма",
                   save_path: Optional[str] = None, show: bool = False, dpi: int = 150) -> None:
    """Отрисовать BoW-гистограмму (частоты визуальных слов) и сохранить её в файл.

    :param histogram: вектор длины V (число визуальных слов)
    :param title: заголовок графика
    :param save_path: путь к файлу PNG (если None — график не сохраняется)
    :param show: показать окно matplotlib (в headless-режиме игнорируется)
    :param dpi: разрешение сохраняемого файла
    """
    hist = np.asarray(histogram, dtype=float)
    words = np.arange(len(hist))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(words, hist, width=1.0, color=BAR_COLOR, edgecolor="none")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Индекс визуального слова (кластер K-средних)")
    ax.set_ylabel("Относительная частота")
    ax.set_xlim(-0.5, max(len(hist) - 0.5, 0))
    ax.grid(axis="y", alpha=0.3)
    if save_path is not None:
        path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        fig.savefig(path, dpi=dpi)
        logger.info("Гистограмма сохранена: %s", path)
    if show:
        plt.show()
    plt.close(fig)


def visualize_image_with_histogram(image_bgr: np.ndarray, histogram: np.ndarray,
                                   predicted_label: Optional[str] = None,
                                   save_path: Optional[str] = None,
                                   show: bool = False, dpi: int = 150) -> None:
    """Совместная визуализация: изображение слева, его BoW-гистограмма справа.

    :param image_bgr: изображение в формате BGR (как возвращает cv2)
    :param histogram: BoW-гистограмма изображения
    :param predicted_label: подпись предсказанного класса (строка) — попадает в заголовок
    :param save_path: путь к файлу PNG (если None — график не сохраняется)
    :param show: показать окно matplotlib
    :param dpi: разрешение сохраняемого файла
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # matplotlib ожидает RGB
    hist = np.asarray(histogram, dtype=float)

    caption = "Входное изображение" if predicted_label is None else f"Входное изображение — класс: {predicted_label}"
    fig, (ax_image, ax_hist) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_image.imshow(rgb)
    ax_image.axis("off")
    ax_image.set_title(caption, fontsize=11)
    ax_hist.bar(np.arange(len(hist)), hist, width=1.0, color=BAR_COLOR)
    ax_hist.set_title("BoW-гистограмма", fontsize=11)
    ax_hist.set_xlabel("Индекс визуального слова")
    ax_hist.set_ylabel("Относительная частота")
    ax_hist.grid(axis="y", alpha=0.3)
    if save_path is not None:
        path = ensure_dir(Path(save_path).parent) / Path(save_path).name
        fig.savefig(path, dpi=dpi)
        logger.info("Совместная визуализация сохранена: %s", path)
    if show:
        plt.show()
    plt.close(fig)
