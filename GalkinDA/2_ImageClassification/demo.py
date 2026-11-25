"""Сохраняет примеры предсказаний для всех классов на одном рисунке."""

import argparse
from pathlib import Path
from textwrap import fill

from main import nonnegative


def main():
    """Выбирает снимки из test, классифицирует их и сохраняет общий рисунок."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Корень датасета')
    parser.add_argument('--model', type=Path, required=True, help='Файл обученной модели')
    parser.add_argument('--weights-dir', type=Path, default=Path('weights'), help='Папка с весами CNN')
    parser.add_argument('--seed', type=nonnegative, default=42, help='Число для повторения случайного выбора')
    parser.add_argument('--output', type=Path, default=Path('results/demo.png'), help='Путь к итоговому PNG')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('--output должен оканчиваться на .png')
    args.device = 'cpu'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from threadpoolctl import threadpool_limits

    from bow import BagOfWords
    from dataset import ArchitectureDatasetLoader, read_image
    from main import make_extractor

    torch.set_num_threads(4)
    try:
        records = ArchitectureDatasetLoader(args.data).load('test', max_images_per_class=1, seed=args.seed)
        model = BagOfWords.load_model(args.model)
        extractor = make_extractor(model.extractor_config, args)
        columns = 4
        rows = (len(records) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(16, 3.5 * rows), layout='constrained')
        for ax in axes.flat:
            ax.axis('off')
        with threadpool_limits(limits=4):
            for ax, (path, true_class) in zip(axes.flat, records):
                image = read_image(path)
                predicted = model.predict(extractor.detect_and_compute(image))
                ax.imshow(image)
                status = 'Верно' if predicted == true_class else 'Ошибка'
                title = f'{status}\n{fill(f"Класс: {true_class}", 36)}\n{fill(f"Ответ: {predicted}", 36)}'
                ax.set_title(title, fontsize=9, color='darkgreen' if status == 'Верно' else 'firebrick')
        fig.suptitle(f'Классификация снимков из тестовой выборки · seed={args.seed}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=140)
        plt.close(fig)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Не удалось создать демонстрацию: {exc}\n')
    print(f'Демонстрация сохранена: {args.output} ({len(records)} изображений)')


if __name__ == '__main__':
    main()
