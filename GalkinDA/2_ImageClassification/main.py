"""Обучает классификатор BoW, проверяет точность и определяет класс фотографии."""
import argparse
import csv
import json
import logging
import os
import pickle
from pathlib import Path
import time


def positive(value):
    """Проверяет, что аргумент — положительное целое число."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Значение должно быть положительным')
    return value


def nonnegative(value):
    """Проверяет, что аргумент — целое число от нуля и выше."""
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('Значение должно быть неотрицательным')
    return value


def cli_argument_parser():
    """Создаёт парсер для режимов train, test и predict."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    modes = {
        'train': 'Обучить модель',
        'test': 'Оценить модель на тестовой выборке',
        'predict': 'Классифицировать одно изображение',
    }
    for mode, description in modes.items():
        p = sub.add_parser(mode, help=description, description=description)
        p.add_argument('--threads', type=positive, default=4)
        p.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
        p.add_argument('--weights-dir', type=Path, default=Path('weights'))
        p.add_argument('--metrics-json', type=Path)
        if mode in {'train', 'test'}:
            p.add_argument('--data', type=Path, required=True)
            p.add_argument('--max-images-per-class', type=nonnegative, default=0)
            p.add_argument('--seed', type=nonnegative, default=42)
        if mode == 'train':
            p.add_argument('--extractor', choices=('sift', 'orb', 'resnet18', 'mobilenetv2'), default='sift')
            p.add_argument('--classifier', choices=('svm', 'rf', 'lr'), default='svm')
            p.add_argument('--C', type=float, default=1.0)
            p.add_argument('--n-words', type=positive, default=128)
            p.add_argument('--n-features', type=positive, default=300)
            p.add_argument('--max-descriptors', type=positive, default=20000)
            p.add_argument('--class-weight', choices=('none', 'balanced'), default='none')
            p.add_argument('--output', type=Path, required=True)
        else:
            p.add_argument('--model', type=Path, required=True)
        if mode == 'test':
            p.add_argument('--predictions', type=Path)
        if mode == 'predict':
            p.add_argument('--image', type=Path, required=True)
            p.add_argument('--save-plot', type=Path)
    return parser


def make_extractor(config, args):
    """Создаёт экстрактор по настройкам обучения."""
    from extractors import FeatureExtractor
    expected_layers = {'resnet18': 'layer4', 'mobilenetv2': 'features.17'}
    if config['name'] in expected_layers:
        layer_matches = config.get('layer') in (None, expected_layers[config['name']])
        if not layer_matches or config.get('input_size', 224) != 224:
            raise ValueError('Модель обучена с неподдерживаемым слоем или размером входа CNN')
    return FeatureExtractor.create(
        config['name'],
        n_features=config['n_features'],
        device=args.device,
        weights_dir=args.weights_dir,
        expected_sha256=config.get('weights_sha256'),
    )


def run_train(args):
    """Обучает модель на trainval и сохраняет её в файл."""
    from bow import BagOfWords
    from dataset import ArchitectureDatasetLoader
    if args.max_descriptors < args.n_words:
        raise ValueError('--max-descriptors должно быть не меньше --n-words')
    records = ArchitectureDatasetLoader(args.data).load('trainval', args.max_images_per_class, args.seed)
    config = {'name': args.extractor, 'n_features': args.n_features}
    if args.extractor in {'resnet18', 'mobilenetv2'}:
        config.update(layer={'resnet18': 'layer4', 'mobilenetv2': 'features.17'}[args.extractor], input_size=224)
    extractor = make_extractor(config, args)
    if hasattr(extractor, 'weights_sha256'):
        config['weights_sha256'] = extractor.weights_sha256
        config['weights'] = 'IMAGENET1K_V1'
    model = BagOfWords(args.n_words, args.classifier, args.seed,
                       None if args.class_weight == 'none' else args.class_weight, C=args.C)
    model.extractor_config = config
    model.fit(records, extractor, args.max_descriptors)
    model.save_model(args.output)
    logging.info('Модель сохранена: %s', args.output)
    return dict(model.metrics, extractor=args.extractor, classifier=args.classifier,
                n_words=args.n_words, max_descriptors=args.max_descriptors,
                seed=args.seed, class_weight=args.class_weight)


def run_test(args):
    """Оценивает сохранённую модель на тестовой выборке."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from bow import BagOfWords
    from dataset import ArchitectureDatasetLoader, read_image
    model = BagOfWords.load_model(args.model)
    extractor = make_extractor(model.extractor_config, args)
    records = ArchitectureDatasetLoader(args.data).load('test', args.max_images_per_class, args.seed)
    unknown = {label for _, label in records} - set(model.classes)
    if unknown:
        raise ValueError(f'На тесте есть классы, которых не было при обучении: {sorted(unknown)}')
    predictions, truth = [], []
    extraction_seconds = histogram_seconds = 0.0
    descriptors_count = 0
    for i, (path, label) in enumerate(records):
        started = time.perf_counter()
        descriptors = extractor.detect_and_compute(read_image(path))
        extraction_seconds += time.perf_counter() - started
        descriptors_count += len(descriptors)
        started = time.perf_counter()
        histogram = model.histogram(descriptors)
        histogram_seconds += time.perf_counter() - started
        predictions.append(model.classifier.predict(histogram[None]).ravel()[0])
        truth.append(label)
        if (i + 1) % 250 == 0:
            logging.info('Проверено изображений: %d/%d', i+1, len(records))
    metrics = {'accuracy': accuracy_score(truth, predictions),
               'macro_f1': f1_score(truth, predictions, labels=model.classes, average='macro', zero_division=0),
               'images': len(records), 'descriptors': descriptors_count,
               'extraction_seconds': extraction_seconds, 'histogram_seconds': histogram_seconds,
               'report': classification_report(truth, predictions, labels=model.classes, zero_division=0, output_dict=True)}
    logging.info('Точность: %.4f, macro F1: %.4f. Дескрипторов: %d, извлечение: %.2f с, гистограммы: %.2f с',
                 metrics['accuracy'], metrics['macro_f1'], descriptors_count, extraction_seconds, histogram_seconds)
    print(classification_report(truth, predictions, labels=model.classes, zero_division=0))
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'true_class', 'predicted_class'])
            writer.writerows((p.relative_to(args.data).as_posix(), c, predicted)
                             for (p, c), predicted in zip(records, predictions))
    return metrics


def run_predict(args):
    """Предсказывает класс одного изображения и показывает его гистограмму."""
    from bow import BagOfWords
    from dataset import read_image
    from visualization import visualize_image_with_histogram
    model = BagOfWords.load_model(args.model)
    image = read_image(args.image)
    descriptors = make_extractor(model.extractor_config, args).detect_and_compute(image)
    histogram = model.histogram(descriptors)
    label = model.classifier.predict(histogram[None]).ravel()[0]
    print(label)
    visualize_image_with_histogram(image, histogram, label, args.save_plot)
    return {'prediction': str(label), 'descriptors': len(descriptors)}


def main():
    """Настраивает запуск и выполняет выбранный режим."""
    parser = cli_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
    started = time.perf_counter()
    try:
        import cv2
        import torch
        from threadpoolctl import threadpool_limits
        cv2.setNumThreads(args.threads)
        torch.set_num_threads(args.threads)
        if args.mode == 'predict' and args.save_plot:
            import matplotlib
            matplotlib.use('Agg')
        with threadpool_limits(limits=args.threads):
            metrics = {'train': run_train, 'test': run_test, 'predict': run_predict}[args.mode](args)
        metrics['total_seconds'] = time.perf_counter() - started
        logging.info('Общее время: %.2f с', metrics['total_seconds'])
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    except (
        OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError,
        EOFError, AttributeError, pickle.UnpicklingError,
    ) as exc:
        logging.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
