"""Показывает фотографии и гистограммы визуальных слов."""
from pathlib import Path


def _finish(fig, save_path):
    """Сохраняет рисунок или показывает его в окне."""
    import matplotlib.pyplot as plt
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_histogram(histogram, save_path=None, ax=None):
    """Рисует гистограмму отдельно или на переданных осях."""
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(histogram)), histogram, width=1)
    ax.set(xlabel='Номер визуального слова', ylabel='Частота', xlim=(-1, len(histogram)))
    if own:
        _finish(fig, save_path)
    return ax


def visualize_image_with_histogram(image, histogram, label, save_path=None):
    """Показывает фотографию, предсказанный класс и гистограмму рядом."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(f'Предсказание: {label}', fontsize=10)
    plot_histogram(histogram, ax=axes[1])
    axes[1].set_title('Мешок визуальных слов')
    _finish(fig, save_path)
