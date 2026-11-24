"""
    train   - построение словаря, обучение классификатора и сохранение модели
    test    - оценка точности сохраненной модели на тестовой выборке
    predict - классификация одного изображения и визуализация BoW-гистограммы
"""

import argparse
import csv
import logging
import os
import sys
import time

import joblib
import numpy as np

from bag_of_words import BagOfWords
from data_loader import LandmarksDataLoader
from feature_extractors import FeatureExtractor
from utils import read_image
import visualization

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "Dataset")

logger = logging.getLogger("bow")


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Классификация изображений методом «мешок визуальных слов»")
    parser.add_argument("--mode", "-m", required=True, choices=["train", "test", "predict"],
                        help="Режим работы")
    parser.add_argument("--extractor", "-e", default="sift",
                        choices=FeatureExtractor.available_names(),
                        help="Метод извлечения признаков (train)")
    parser.add_argument("--layer",
                        help="Узел CNN, например features.7 для AlexNet или layer2 для ResNet-101")
    parser.add_argument("--vocab-size", "-k", type=int, default=300,
                        help="Размер словаря визуальных слов (train)")
    parser.add_argument("--max-descriptors", type=int, default=150000,
                        help="Максимум дескрипторов для кластеризации K-средних (train)")
    parser.add_argument("--data-root", "-d", default=DEFAULT_DATA_ROOT,
                        help="Корневая директория набора данных (train, test)")
    parser.add_argument("--sources", nargs="+", default=list(LandmarksDataLoader.SOURCES),
                        choices=list(LandmarksDataLoader.SOURCES),
                        help="Используемые части набора данных (train, test)")
    parser.add_argument("--model-path", "-p", default=None,
                        help="Файл модели; по умолчанию вычисляется по экстрактору, слою и K")
    parser.add_argument("--image", "-i", default=None,
                        help="Путь к изображению (predict)")
    parser.add_argument("--output-dir", "-o", default=os.path.join(SCRIPT_DIR, "results"),
                        help="Директория для результатов (CSV, графики)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Устройство для нейросетевых моделей")
    parser.add_argument("--top-k", type=int, default=5,
                        help="K для точности top-K (test) и числа выводимых классов (predict)")
    parser.add_argument("--cache-dir", default=None,
                        help="Директория для кэша дескрипторов")
    parser.add_argument("--show", action="store_true",
                        help="Показать графики в окне (predict)")
    return parser.parse_args()


def extractor_tag(name, layer):
    """Имя экстрактора для модели и кэша с учетом нестандартного слоя."""
    resolved = FeatureExtractor.resolve_layer(name, layer)
    if resolved == FeatureExtractor.resolve_layer(name):
        return name
    return f"{name}__{resolved}"


def default_model_path(args):
    tag = extractor_tag(args.extractor, args.layer)
    file_name = f"{tag}_rf_k{args.vocab_size}.pkl"
    return os.path.join(SCRIPT_DIR, "models", file_name)

# ---------------------------------------------------------------------------
# Извлечение дескрипторов для набора изображений
# ---------------------------------------------------------------------------

def extract_descriptors_for_dataset(extractor, paths):
    descriptor_list = []
    start_time = time.time()
    for index, path in enumerate(paths):
        try:
            image = read_image(path)
            descriptors = extractor.detect_and_compute(image)
        except (FileNotFoundError, ValueError) as error:
            logger.warning("Пропуск изображения: %s", error)
            descriptors = extractor.empty_descriptors()
        descriptor_list.append(descriptors.astype(np.float16))

        if (index + 1) % 200 == 0 or index + 1 == len(paths):
            elapsed = time.time() - start_time
            logger.info("Обработано изображений: %d / %d (%.1f с)", index + 1, len(paths), elapsed)
    return descriptor_list


def load_or_extract_descriptors(args, extractor_name, split, paths, device, *, layer=None):
    """Кэширование дескрипторов."""
    cache_path = None
    if args.cache_dir is not None:
        os.makedirs(args.cache_dir, exist_ok=True)
        sources_tag = "_".join(sorted(args.sources))
        tag = extractor_tag(extractor_name, layer)
        cache_path = os.path.join(args.cache_dir, f"{tag}_{split}_{sources_tag}.joblib")
        if os.path.isfile(cache_path):
            cached = joblib.load(cache_path)
            if cached["paths"] == paths:
                logger.info("Дескрипторы загружены из кэша: %s", cache_path)
                return cached["descriptors"]
            logger.warning("Кэш %s не соответствует набору изображений, пересчет", cache_path)

    logger.info("Создание экстрактора признаков: %s", extractor_name)
    extractor = FeatureExtractor.create(extractor_name, device=device, layer=layer)
    descriptor_list = extract_descriptors_for_dataset(extractor, paths)

    if cache_path is not None:
        joblib.dump({"paths": paths, "descriptors": descriptor_list}, cache_path)
        logger.info("Дескрипторы сохранены в кэш: %s", cache_path)
    return descriptor_list


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def top_k_accuracy(scores, labels, k):
    # scores - изображения x классы (матрица достоверностей)
    # labels - классы из разметки
    labels = np.asarray(labels)
    k = min(k, scores.shape[1])
    top_classes = np.argsort(-scores, axis=1)[:, :k]
    hits = 0
    for row in range(len(labels)):
        if labels[row] in top_classes[row]:
            hits += 1
    return hits / len(labels)


def log_per_class_accuracy(labels, predictions, class_names):
    """Логгировние доли верных предсказаний для каждого класса."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    logger.info("Точность по классам:")
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float((predictions[mask] == class_id).mean())
        logger.info("  %-42s %5.3f  (%d изобр.)", class_name, accuracy, count)


# ---------------------------------------------------------------------------
# Режимы работы
# ---------------------------------------------------------------------------

def run_train(args):
    loader = LandmarksDataLoader(args.data_root, args.sources)
    paths, labels = loader.load("trainval")
    logger.info("Тренировочная выборка: %d изображений, %d классов",
                len(paths), len(loader.class_names))

    layer = FeatureExtractor.resolve_layer(args.extractor, args.layer)
    descriptor_list = load_or_extract_descriptors(args, args.extractor, "trainval",
                                                  paths, args.device, layer=layer)

    model = BagOfWords(args.extractor, loader.class_names, args.vocab_size,
                       max_descriptors=args.max_descriptors, layer=layer)

    logger.info("Построение словаря (K-средних, K = %d)", args.vocab_size)
    start_time = time.time()
    model.build_vocabulary(descriptor_list)
    logger.info("Словарь построен за %.1f с", time.time() - start_time)

    logger.info("Вычисление BoW-гистограмм")
    histograms = model.compute_histograms(descriptor_list)

    logger.info("Обучение случайного леса")
    model.train(histograms, labels)
    train_accuracy = float((model.predict(histograms) == np.asarray(labels)).mean())
    logger.info("Точность на тренировочной выборке: %.4f", train_accuracy)

    model_path = args.model_path or default_model_path(args)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(model_path)
    logger.info("Модель сохранена: %s", model_path)


def load_trained_model(args):
    model_path = args.model_path or default_model_path(args)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}. "
                                f"Сначала обучите модель в режиме train")
    model = BagOfWords.load_model(model_path)
    layer = FeatureExtractor.resolve_layer(model.extractor_name, model.layer)
    if args.layer is not None and args.layer != layer:
        raise ValueError(f"Слой {args.layer!r} не совпадает со слоем модели {layer!r}")
    logger.info("Загружена модель %s (экстрактор %s, слой %s, K = %d)",
                model_path, model.extractor_name, layer or "—", model.vocab_size)
    return model, model_path


def run_test(args):
    model, model_path = load_trained_model(args)

    loader = LandmarksDataLoader(args.data_root, args.sources)
    if loader.class_names != model.class_names:
        raise ValueError("Классы набора данных не совпадают с классами модели")
    paths, labels = loader.load("test")
    logger.info("Тестовая выборка: %d изображений", len(paths))

    descriptor_list = load_or_extract_descriptors(args, model.extractor_name, "test",
                                                  paths, args.device, layer=model.layer)
    histograms = model.compute_histograms(descriptor_list)

    scores = model.predict_scores(histograms)
    predictions = np.argmax(scores, axis=1)
    top1 = top_k_accuracy(scores, labels, 1)
    topk = top_k_accuracy(scores, labels, args.top_k)

    log_per_class_accuracy(labels, predictions, model.class_names)
    logger.info("Точность top-1: %.4f", top1)
    logger.info("Точность top-%d: %.4f", args.top_k, topk)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = os.path.join(args.output_dir, f"test_predictions_{model_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "true_class", "predicted_class", "correct"])
        for path, label, prediction in zip(paths, labels, predictions):
            writer.writerow([os.path.relpath(path, args.data_root),
                             model.class_names[label],
                             model.class_names[prediction],
                             int(label == prediction)])
    logger.info("Результаты классификации сохранены: %s", csv_path)

    print(f"Accuracy top-1: {top1:.4f}")
    print(f"Accuracy top-{args.top_k}: {topk:.4f}")


def run_predict(args):
    if args.image is None:
        raise ValueError("Для режима predict необходимо указать --image")
    image = read_image(args.image)

    model, model_path = load_trained_model(args)
    extractor = FeatureExtractor.create(model.extractor_name, device=args.device, layer=model.layer)
    descriptors = extractor.detect_and_compute(image)
    logger.info("Найдено дескрипторов: %d", len(descriptors))
    if len(descriptors) == 0:
        logger.warning("На изображении не найдено ни одного дескриптора, "
                       "гистограмма будет нулевой")

    histogram = model.compute_histogram(descriptors)
    scores = model.predict_scores(histogram.reshape(1, -1))[0]
    ranking = np.argsort(-scores)

    predicted_class = model.class_names[ranking[0]]
    print(f"Предсказанный класс: {predicted_class}")
    print(f"Top-{args.top_k} классов (достоверность):")
    for position in range(min(args.top_k, len(ranking))):
        class_id = ranking[position]
        print(f"  {position + 1}. {model.class_names[class_id]} ({scores[class_id]:.3f})")

    if not args.show:
        visualization.use_non_interactive_backend()
    os.makedirs(args.output_dir, exist_ok=True)
    image_tag = os.path.splitext(os.path.basename(args.image))[0]
    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    histogram_path = os.path.join(args.output_dir, f"{image_tag}_{model_tag}_hist.png")
    combined_path = os.path.join(args.output_dir, f"{image_tag}_{model_tag}_result.png")

    visualization.plot_histogram(histogram, f"BoW-гистограмма ({model.extractor_name}, "
                                            f"K = {model.vocab_size})",
                                 save_path=histogram_path, show=args.show)
    visualization.visualize_image_with_histogram(image, histogram, predicted_class,
                                                 save_path=combined_path, show=args.show)
    logger.info("Графики сохранены: %s, %s", histogram_path, combined_path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    args = cli_argument_parser()

    if args.vocab_size < 2:
        logger.error("Размер словаря должен быть не меньше 2")
        sys.exit(2)
    if args.top_k < 1:
        logger.error("Значение --top-k должно быть положительным")
        sys.exit(2)

    try:
        if args.mode == "train":
            run_train(args)
        elif args.mode == "test":
            run_test(args)
        else:
            run_predict(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.error("Выполнение прервано пользователем")
        sys.exit(130)


if __name__ == "__main__":
    main()
