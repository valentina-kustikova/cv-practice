import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from bow import CLASSIFIERS, BagOfWords
from data import DATA_LOADERS, read_image
from extractors import CLASSICAL, EXTRACTORS, FeatureExtractor
from visualization import plot_histogram, visualize_image_with_histogram

logger = logging.getLogger('bow')
LOG_EVERY = 500 #как часто писать в лог прогресс обработки изображений


def cli_argument_parser():
    parser = argparse.ArgumentParser(description='Классификация изображений методом мешка визуальных слов')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    modes = parser.add_subparsers(dest='mode', required=True)

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument('-d', '--data', required=True, help='корневая директория набора данных')
    data.add_argument('--dataset', default='landmarks', choices=DATA_LOADERS, help='формат набора данных')

    train = modes.add_parser('train', parents=[data], help='обучение модели')
    train.add_argument('-e', '--extractor', required=True, choices=EXTRACTORS, help='метод извлечения признаков')
    train.add_argument('-m', '--model', required=True, help='путь для сохранения модели')
    train.add_argument('-k', '--vocab-size', type=int, default=1000, help='число визуальных слов')
    train.add_argument('-c', '--classifier', default='linear_svm', choices=CLASSIFIERS, help='классификатор')
    train.add_argument('--vocab-images', type=int, default=500,
                       help='число случайных обучающих изображений для построения словаря')
    train.add_argument('--max-descriptors', type=int, default=200000, help='максимум дескрипторов для K-средних')
    train.add_argument('--pca', type=int, help='размерность PCA перед K-средних, 0 отключает '
                                                '(по умолчанию 128 для CNN, без PCA для SIFT и ORB)')
    train.add_argument('--seed', type=int, default=42, help='зерно генератора случайных чисел')
    params = train.add_argument_group('параметры экстракторов')
    params.add_argument('--image-size', type=int, help='SIFT, ORB: изображения с большей стороной длиннее этого '
                                                       'размера уменьшаются с сохранением пропорций (640)')
    params.add_argument('--nfeatures', type=int, help='SIFT, ORB: максимум ключевых точек (1000)')
    params.add_argument('--layer', help='CNN: слой сети, выход которого берется как карта признаков '
                                        '(layer4 для resnet50, features.8 для efficientnet_b0)')
    params.add_argument('--input-size', type=int, help='CNN: размер меньшей стороны входа сети (288)')

    test = modes.add_parser('test', parents=[data], help='тестирование модели')
    test.add_argument('-m', '--model', required=True, help='файл обученной модели')
    test.add_argument('-o', '--output', help='CSV-файл с результатами классификации тестовых изображений')

    predict = modes.add_parser('predict', help='классификация одного изображения')
    predict.add_argument('-m', '--model', required=True, help='файл обученной модели')
    predict.add_argument('-i', '--image', required=True, help='путь к изображению')
    predict.add_argument('-s', '--save-dir', help='директория для сохранения графиков')
    predict.add_argument('--no-show', action='store_true', help='не показывать окно с результатом')
    return parser.parse_args()


def extract_all(extractor, paths, handle):
    """Извлекает дескрипторы каждого изображения и передает их в handle(i, descriptors)."""
    start = time.time()
    for i, path in enumerate(paths):
        handle(i, extractor.detect_and_compute(read_image(path)))
        if (i + 1) % LOG_EVERY == 0 or i + 1 == len(paths):
            logger.info('  обработано %d/%d изображений (%.0f с)', i + 1, len(paths), time.time() - start)


def compute_histograms(model, paths):
    histograms = np.zeros((len(paths), model.vocab_size), dtype=np.float32)

    def store(i, descriptors):
        histograms[i] = model.compute_histogram(descriptors)

    extract_all(model.extractor, paths, store)
    return histograms


def extractor_params(args):
    """Параметры экстрактора из аргументов командной строки: для классических методов и для CNN свои."""
    if args.extractor in CLASSICAL:
        return {'image_size': args.image_size, 'nfeatures': args.nfeatures}
    return {'layer': args.layer, 'input_size': args.input_size}


def run_train(args):
    """Словарь K-средних по подвыборке изображений -> гистограммы обучающей выборки -> классификатор."""
    if args.vocab_images <= 0 or args.max_descriptors <= 0:
        raise ValueError('--vocab-images и --max-descriptors должны быть положительными')
    if args.pca is not None and args.pca < 0:
        raise ValueError('--pca не может быть отрицательным')
    loader = DATA_LOADERS[args.dataset](args.data)
    paths, labels = loader.load('train')
    extractor = FeatureExtractor.create(args.extractor, **extractor_params(args))
    pca = args.pca if args.pca is not None else (0 if args.extractor in CLASSICAL else 128)
    model = BagOfWords(extractor, args.vocab_size, args.classifier, pca or None, args.seed)

    rng = np.random.default_rng(args.seed)
    vocab_paths = [paths[i] for i in rng.choice(len(paths), min(args.vocab_images, len(paths)), replace=False)]
    per_image = max(1, args.max_descriptors // len(vocab_paths)) #одинаковый вклад каждого изображения в словарь
    sampled = []

    def collect(_, descriptors):
        if len(descriptors) > per_image:
            descriptors = descriptors[rng.choice(len(descriptors), per_image, replace=False)]
        sampled.append(descriptors)

    logger.info('Шаг 1/3: дескрипторы для словаря (%d изображений, метод %s)', len(vocab_paths), extractor.name)
    extract_all(extractor, vocab_paths, collect)
    model.build_vocabulary(np.vstack(sampled))
    sampled.clear() #дескрипторы для словаря больше не нужны

    logger.info('Шаг 2/3: BoW-гистограммы обучающей выборки (%d изображений)', len(paths))
    histograms = compute_histograms(model, paths)

    logger.info('Шаг 3/3: обучение классификатора')
    model.train(histograms, labels, loader.class_names)
    logger.info('Точность на обучающей выборке: %.4f', accuracy_score(labels, model.predict_histograms(histograms)))

    folder = os.path.dirname(args.model)
    if folder:
        os.makedirs(folder, exist_ok=True)
    model.save_model(args.model)


def run_test(args):
    """Точность, отчет по классам и матрица ошибок на тестовой выборке, при необходимости CSV с ответами."""
    model = BagOfWords.load_model(args.model)
    loader = DATA_LOADERS[args.dataset](args.data)
    if loader.class_names != model.class_names:
        raise ValueError(f'Классы набора данных {loader.class_names} не совпадают с классами модели {model.class_names}')
    paths, labels = loader.load('test')

    logger.info('BoW-гистограммы тестовой выборки (%d изображений)', len(paths))
    predictions = model.predict_histograms(compute_histograms(model, paths))
    accuracy = accuracy_score(labels, predictions)
    logger.info('Точность на тестовой выборке: %.4f', accuracy)

    classes = range(len(model.class_names))
    print(f'\nAccuracy: {accuracy:.4f}\n')
    print(classification_report(labels, predictions, labels=classes, target_names=model.class_names,
                                digits=4, zero_division=0))
    print('Матрица ошибок (строки: истинный класс, столбцы: предсказанный):')
    print(confusion_matrix(labels, predictions, labels=classes))

    if args.output:
        folder = os.path.dirname(args.output)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'true', 'predicted'])
            for path, true, predicted in zip(paths, labels, predictions):
                writer.writerow([os.path.relpath(path, args.data), model.class_names[true],
                                 model.class_names[predicted]])
        logger.info('Результаты классификации сохранены: %s', args.output)


def run_predict(args):
    """Класс одного изображения, его BoW-гистограмма и визуализация."""
    model = BagOfWords.load_model(args.model)
    image = read_image(args.image)
    label, histogram = model.predict(image)
    print(f'Predicted class: {label}')

    hist_path = joint_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.image))[0]
        hist_path = os.path.join(args.save_dir, f'{name}_histogram.png')
        joint_path = os.path.join(args.save_dir, f'{name}_with_histogram.png')
        plot_histogram(histogram, f'BoW-гистограмма ({model.extractor.name}, K = {model.vocab_size})',
                       save_path=hist_path, show=False)
    visualize_image_with_histogram(image, histogram, label, save_path=joint_path, show=not args.no_show)
    if args.save_dir:
        logger.info('Графики сохранены: %s, %s', hist_path, joint_path)


def main():
    args = cli_argument_parser()
    logging.basicConfig(level=args.log_level, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    modes = {'train': run_train, 'test': run_test, 'predict': run_predict}
    try:
        logger.info('Режим: %s', args.mode)
        modes[args.mode](args)
    except (OSError, ValueError, RuntimeError) as e:
        logger.error('%s', e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning('Прервано пользователем')
        sys.exit(130)
    except Exception:
        logger.exception('Непредвиденная ошибка')
        sys.exit(1)


if __name__ == '__main__':
    main()
