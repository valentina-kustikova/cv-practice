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
    # Рисуем в масштабе показа, иначе тонкие линии на больших фото
    # исчезают при уменьшении matplotlib и сохранении общей фигуры.
    canvas = image.copy()
    longest = max(canvas.shape[:2])
    scale = min(900, max(450, longest)) / longest
    if scale != 1:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    display_keypoints = []
    if keypoints:
        sx, sy = canvas.shape[1] / image.shape[1], canvas.shape[0] / image.shape[0]
        display_keypoints = [cv2.KeyPoint(kp.pt[0] * sx, kp.pt[1] * sy,
                                        kp.size * min(sx, sy), kp.angle,
                                        kp.response, kp.octave, kp.class_id) for kp in keypoints]

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
    # Поверх панели и фотографии: маркеры не затемняются гистограммой.
    oriented = sorted((kp for kp in display_keypoints if kp.angle >= 0),
                      key=lambda kp: kp.response, reverse=True)[:30]
    for kp in oriented:
        center = tuple(round(v) for v in kp.pt)
        radius = max(1, round(kp.size / 2))
        angle = np.deg2rad(kp.angle)
        # Минимум длины относится только к указателю направления, не к масштабу.
        length = max(14, radius)
        end = (round(kp.pt[0] + length * np.cos(angle)),
               round(kp.pt[1] + length * np.sin(angle)))
        for color, thickness in [((0, 0, 0), 4), ((0, 255, 0), 2)]:
            cv2.circle(canvas, center, radius, color, thickness, cv2.LINE_AA)
            cv2.arrowedLine(canvas, center, end, color, thickness, cv2.LINE_AA,
                            tipLength=0.3)
        cv2.circle(canvas, center, 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 2, (0, 255, 255), -1, cv2.LINE_AA)
    for kp in display_keypoints:
        if kp.angle < 0:
            center = tuple(round(v) for v in kp.pt)
            cv2.circle(canvas, center, 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 2, (0, 255, 0), -1, cv2.LINE_AA)
    return canvas


def visualize_image_with_histogram(image, histogram, label=None, keypoints=None, n_descriptors=None,
                                   save_path=None, show=True):
    """Совместная визуализация: изображение с наложенной гистограммой и отдельный график гистограммы.

    n_descriptors – число дескрипторов изображения: столько слов распределено по столбцам гистограммы.
    """
    overlay = draw_histogram_overlay(image, histogram, label, keypoints)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})
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
