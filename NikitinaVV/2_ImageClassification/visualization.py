import cv2
import matplotlib.pyplot as plt
import numpy as np

TOP_WORDS = 5 #сколько самых частых слов выделять цветом
COLORS = plt.get_cmap('tab10').colors
BAR_COLOR = '#9aa5b1'


def top_words(histogram, count=TOP_WORDS):
    """Номера самых частых слов гистограммы (только встретившиеся), по убыванию частоты."""
    order = np.argsort(histogram)[::-1][:count]
    return [int(k) for k in order if histogram[k] > 0]


def draw_bars(ax, histogram):
    """Столбцы гистограммы: самые частые слова цветные и подписаны в легенде, остальные серые."""
    ax.bar(np.arange(len(histogram)), histogram, width=1.0, color=BAR_COLOR)
    for color, k in zip(COLORS, top_words(histogram)):
        ax.bar(k, histogram[k], width=1.0, color=color, label=f'слово {k}')
    if histogram.any():
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.set_xlim(-0.5, len(histogram) - 0.5)
    ax.set_xlabel('Номер визуального слова')
    ax.set_ylabel('Доля дескрипторов')


def finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_histogram(histogram, title='BoW-гистограмма', save_path=None, show=True):
    """Рисует BoW-гистограмму, при необходимости сохраняет ее в файл."""
    fig, ax = plt.subplots(figsize=(10, 4))
    draw_bars(ax, histogram)
    ax.set_title(title)
    finish(fig, save_path, show)


def visualize_image_with_histogram(image, histogram, label, save_path=None, show=True):
    """Слева изображение с предсказанным классом, справа его BoW-гистограмма."""
    fig, (ax_image, ax_hist) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={'width_ratios': [1, 1.4]})
    ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_image.set_title(f'Предсказанный класс: {label}')
    ax_image.axis('off')

    draw_bars(ax_hist, histogram)
    used = int(np.count_nonzero(histogram))
    ax_hist.set_title(f'BoW-гистограмма: встретилось {used} из {len(histogram)} слов')
    finish(fig, save_path, show)
