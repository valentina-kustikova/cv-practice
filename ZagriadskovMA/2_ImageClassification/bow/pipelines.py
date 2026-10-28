"""Пайплайны режимов запуска: train, test, predict."""

import csv
import inspect
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .bow import BagOfWords
from .data import DATA_LOADERS
from .extractors import FeatureExtractor
from .utils import read_image
from .visualization import plot_histogram, visualize_image_with_histogram

logger = logging.getLogger(__name__)

LOG_EVERY = 500


def _extract_all(extractor, paths, callback):
    """Извлекает дескрипторы для каждого изображения и передает их в callback(i, descriptors)."""
    start = time.time()
    for i, path in enumerate(paths):
        _, descriptors = extractor.detect_and_compute(read_image(path))
        callback(i, descriptors)
        if (i + 1) % LOG_EVERY == 0 or i + 1 == len(paths):
            logger.info("  обработано %d/%d изображений (%.1f c)", i + 1, len(paths), time.time() - start)


def _compute_histograms(model, paths):
    histograms = np.zeros((len(paths), model.vocab_size), dtype=np.float32)

    def store(i, descriptors):
        histograms[i] = model.compute_histogram(descriptors)

    _extract_all(model.extractor, paths, store)
    return histograms


def run_train(args):
    """Обучение: словарь K-средних по подвыборке изображений -> гистограммы -> классификатор."""
    loader = DATA_LOADERS[args.dataset](args.data_root, limit_per_class=args.limit_per_class)
    samples = loader.get_samples("train")
    paths = [p for p, _ in samples]
    labels = np.array([label for _, label in samples])

    extractor = FeatureExtractor.create(args.extractor, **_extractor_params(args))
    pca_dim = args.pca_dim if args.pca_dim is not None else extractor.default_pca_dim
    model = BagOfWords(extractor, vocab_size=args.vocab_size, classifier=args.classifier,
                       pca_dim=pca_dim or None, max_descriptors=args.max_descriptors,
                       tfidf=args.tfidf, random_state=args.seed)

    # 1. Словарь строится по случайной подвыборке изображений, чтобы не хранить
    #    в памяти дескрипторы всего набора данных
    rng = np.random.default_rng(args.seed)
    n_vocab = min(args.vocab_images, len(paths))
    vocab_idx = rng.choice(len(paths), n_vocab, replace=False)
    per_image = max(1, args.max_descriptors // n_vocab)
    sampled = []

    def collect(_, descriptors):
        if len(descriptors) > per_image:
            descriptors = descriptors[rng.choice(len(descriptors), per_image, replace=False)]
        sampled.append(descriptors)

    logger.info("Шаг 1/3: дескрипторы для словаря (%d изображений, метод %s)", n_vocab, extractor.name)
    _extract_all(extractor, [paths[i] for i in vocab_idx], collect)
    model.build_vocabulary(sampled)
    del sampled

    # 2. BoW-гистограммы всех обучающих изображений
    logger.info("Шаг 2/3: BoW-гистограммы обучающей выборки (%d изображений)", len(paths))
    histograms = _compute_histograms(model, paths)

    # 3. Классификатор
    logger.info("Шаг 3/3: обучение классификатора")
    histograms = model.fit_classifier(histograms, labels, loader.class_names)
    train_acc = accuracy_score(labels, model.predict_histograms(histograms))
    logger.info("Точность на обучающей выборке: %.4f", train_acc)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model)
    return model


def _extractor_params(args):
    """Отбирает из аргументов командной строки параметры конструктора выбранного экстрактора."""
    extractor_cls = FeatureExtractor._registry[args.extractor]
    accepted = inspect.signature(extractor_cls.__init__).parameters
    return {name: value for name, value in vars(args).items()
            if name in accepted and name != "self" and value is not None}


def run_test(args):
    """Тестирование: точность, отчет по классам, матрица ошибок, опционально CSV с результатами."""
    model = BagOfWords.load_model(args.model, **_device_override(args))
    loader = DATA_LOADERS[args.dataset](args.data_root, limit_per_class=args.limit_per_class)
    if list(loader.class_names) != list(model.class_names):
        raise ValueError(f"Классы набора данных {loader.class_names} не совпадают с классами модели {model.class_names}")

    samples = loader.get_samples("test")
    paths = [p for p, _ in samples]
    labels = np.array([label for _, label in samples])

    logger.info("BoW-гистограммы тестовой выборки (%d изображений)", len(paths))
    histograms = _compute_histograms(model, paths)
    predictions = model.predict_histograms(histograms)

    accuracy = accuracy_score(labels, predictions)
    logger.info("Точность на тестовой выборке: %.4f", accuracy)
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(labels, predictions, labels=range(len(model.class_names)),
                                target_names=model.class_names, digits=4, zero_division=0))
    print("Матрица ошибок (строки – истинный класс, столбцы – предсказанный):")
    print(confusion_matrix(labels, predictions, labels=range(len(model.class_names))))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "true", "predicted"])
            for path, true, pred in zip(paths, labels, predictions):
                writer.writerow([str(path), model.class_names[true], model.class_names[pred]])
        logger.info("Результаты классификации сохранены: %s", args.output)
    return accuracy


def run_predict(args):
    """Классификация одного изображения с визуализацией BoW-гистограммы."""
    model = BagOfWords.load_model(args.model, **_device_override(args))
    image = read_image(args.image)
    label, histogram, keypoints = model.predict(image)
    logger.info("Изображение %s: предсказанный класс '%s'", args.image, label)
    print(f"Predicted class: {label}")

    hist_path = overlay_path = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        hist_path = save_dir / f"{stem}_histogram.png"
        overlay_path = save_dir / f"{stem}_with_histogram.png"

    # Отдельный график гистограммы только сохраняется в файл, на экран выводится одно совместное окно
    if hist_path:
        plot_histogram(histogram, title=f"BoW-гистограмма ({model.extractor.name}, K={model.vocab_size})",
                       save_path=hist_path, show=False)
    visualize_image_with_histogram(image, histogram, label=label, n_descriptors=len(keypoints),
                                   keypoints=keypoints if args.draw_keypoints else None,
                                   save_path=overlay_path, show=not args.no_show)
    return label


def _device_override(args):
    device = getattr(args, "device", None)
    return {"device": device} if device else {}
