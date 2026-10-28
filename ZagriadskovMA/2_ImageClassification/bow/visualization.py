"""Отображение BoW-гистограмм и результатов классификации."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(histogram, title="BoW-гистограмма", save_path=None, show=True):
    """Рисует BoW-гистограмму (доля каждого визуального слова) с возможностью сохранения в файл."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(histogram)), histogram, width=1.0, color="#3b6fb6")
    ax.set_xlabel("Номер визуального слова")
    ax.set_ylabel("Частота")
    ax.set_title(title)
    ax.set_xlim(-0.5, len(histogram) - 0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def draw_histogram_overlay(image, histogram, label=None, keypoints=None, height_ratio=0.3):
    """Накладывает BoW-гистограмму на изображение средствами OpenCV.

    Внизу изображения рисуется полупрозрачная панель со столбцами гистограммы,
    сверху – предсказанный класс; при наличии ключевых точек они тоже отрисовываются.
    """
    canvas = image.copy()
    if max(canvas.shape[:2]) < 450:
        scale = 450 / max(canvas.shape[:2])
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if keypoints:
            keypoints = [cv2.KeyPoint(kp.pt[0] * scale, kp.pt[1] * scale, kp.size * scale) for kp in keypoints]
    if keypoints:
        canvas = cv2.drawKeypoints(canvas, keypoints, None, color=(0, 255, 0))

    h, w = canvas.shape[:2]
    panel_h = int(h * height_ratio)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), thickness=-1)
    peak = histogram.max() if histogram.max() > 0 else 1.0
    bar_w = w / len(histogram)
    for i, value in enumerate(histogram):
        bar_h = int((value / peak) * (panel_h - 6))
        x0, x1 = int(i * bar_w), max(int((i + 1) * bar_w) - 1, int(i * bar_w))
        cv2.rectangle(overlay, (x0, h - 3 - bar_h), (x1, h - 3), (0, 200, 255), thickness=-1)
    canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

    if label is not None:
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def visualize_image_with_histogram(image, histogram, label=None, keypoints=None, n_descriptors=None,
                                   save_path=None, show=True):
    """Совместная визуализация: изображение с наложенной гистограммой и отдельный график гистограммы.

    n_descriptors – число дескрипторов изображения: столько слов распределено по столбцам гистограммы.
    """
    overlay = draw_histogram_overlay(image, histogram, label, keypoints)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    ax_img.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Предсказанный класс: {label}" if label else "Изображение")
    ax_img.axis("off")
    ax_hist.bar(np.arange(len(histogram)), histogram, width=1.0, color="#3b6fb6")
    ax_hist.set_xlabel("Номер визуального слова")
    ax_hist.set_ylabel("Доля дескрипторов")
    used = int(np.count_nonzero(histogram))
    suffix = f": {n_descriptors} дескрипторов, {used} из {len(histogram)} слов" if n_descriptors is not None else ""
    ax_hist.set_title(f"BoW-гистограмма{suffix}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return overlay
