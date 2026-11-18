from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .bow import BagOfWords
from .data import DataLoader, FolderDataLoader
from .extractors import FeatureExtractor
from .utils import ensure_dir, load_sample, read_image
from .visualize import plot_histogram, visualize_image_with_histogram

logger = logging.getLogger(__name__)

CLASSICAL_EXTRACTORS = ("sift", "orb")


def create_data_loader(args) -> DataLoader:
    """Создать загрузчик данных по аргументам командной строки (--dataset/--data-root)."""
    if args.dataset != "folder": 
        raise ValueError(
            f"Неизвестный тип набора данных: '{args.dataset}'. "
            f"Допустимо 'folder' или путь к каталогу набора (например --dataset ./Dataset)."
        )
    return FolderDataLoader(args.data_root, max_per_class=args.max_per_class)


def create_extractor(args) -> FeatureExtractor:
    """Создать извлекатель признаков по аргументам командной строки (--extractor)."""
    if args.extractor in CLASSICAL_EXTRACTORS:
        return FeatureExtractor.create(args.extractor, min_side=args.min_side)
    return FeatureExtractor.create(
        args.extractor,
        input_size=args.cnn_size,
        pretrained=not args.no_pretrained,
        device=args.device,
    )


def _label_name(label: int, class_names: Sequence[str]) -> str:
    """Имя класса по индексу; если имена недоступны — строка 'класс_<номер>'."""
    return class_names[label] if label < len(class_names) else f"класс_{label}"


def run_train(args) -> BagOfWords:
    """Режим train: построение словаря визуальных слов (K-средних), BoW-гистограмм
    обучающей выборки, обучение классификатора и сохранение модели в файл."""
    logger.info("=== РЕЖИМ TRAIN ===")
    logger.info(
        "Параметры: экстрактор=%s, набор данных=%s (%s), словарь=%d слов, классификатор=%s, нормировка=%s",
        args.extractor, args.dataset, args.data_root, args.vocab_size, args.classifier, args.hist_norm,
    )

    loader = create_data_loader(args)
    samples = loader.load("trainval")
    if len(samples) < 4:
        raise ValueError("Обучающая выборка слишком мала (нужно хотя бы несколько изображений на класс)")

    extractor = create_extractor(args)
    bow = BagOfWords(extractor, vocab_size=args.vocab_size, classifier_type=args.classifier,
                     hist_norm=args.hist_norm, random_state=args.seed)
    bow.class_names = list(loader.class_names)
    logger.info("Классы (%d): %s", len(bow.class_names), ", ".join(bow.class_names))

    descriptors: List[np.ndarray] = []
    labels: List[int] = []
    skipped_images = 0
    log_every = max(1, len(samples) // 20)
    for index, (source, label) in enumerate(samples, start=1):
        try:
            image = load_sample(source)
        except Exception as exc:
            logger.warning("Изображение '%s' пропущено: %s", source, exc)
            skipped_images += 1
            continue
        image_descriptors = extractor.detect_and_compute(image)
        if args.max_descriptors_per_image and len(image_descriptors) > args.max_descriptors_per_image:
            rng = np.random.default_rng(args.seed + index)
            chosen = rng.choice(len(image_descriptors), size=args.max_descriptors_per_image, replace=False)
            image_descriptors = image_descriptors[chosen]
        if len(image_descriptors) == 0:
            skipped_images += 1
            continue
        descriptors.append(image_descriptors)
        labels.append(label)
        if index % log_every == 0 or index == len(samples):
            logger.info("Извлечение признаков: обработано %d/%d изображений", index, len(samples))
    if skipped_images:
        logger.warning("Изображений пропущено (ошибка чтения или нет ключевых точек): %d", skipped_images)
    if len(descriptors) < 2:
        raise ValueError("Слишком мало изображений с дескрипторами — обучение невозможно")

    bow.build_vocabulary(descriptors, sample_size=args.vocab_sample_size)

    histograms = [bow.compute_histogram(item) for item in descriptors]

    bow.train_classifier(histograms, labels)
    bow.save_model(args.model_path)
    logger.info("Готово: модель сохранена в '%s' (классы: %s)", args.model_path, ", ".join(bow.class_names))
    return bow


def run_test(args) -> float:
    """Режим test: загрузка обученной модели, оценка точности на тестовой выборке.
    """
    logger.info("=== РЕЖИМ TEST ===")
    bow = BagOfWords.load_model(args.model_path)
    logger.info("Параметры экстрактора берутся из файла модели (аргументы --extractor и подобные игнорируются)")

    loader = create_data_loader(args)
    samples = loader.load("test")
    class_names = list(loader.class_names) or list(bow.class_names)
    logger.info("Классы (%d): %s", len(class_names), ", ".join(class_names) if class_names else "н/д")

    y_true: List[int] = []
    y_pred: List[int] = []
    image_names: List[str] = []
    log_every = max(1, len(samples) // 20)
    for index, (source, label) in enumerate(samples, start=1):
        try:
            image = load_sample(source)
        except Exception as exc:
            logger.warning("Изображение '%s' пропущено: %s", source, exc)
            continue
        histogram = bow.histogram_from_image(image)
        y_pred.append(bow.predict_histogram(histogram))
        y_true.append(label)
        image_names.append(str(source))
        if index % log_every == 0 or index == len(samples):
            logger.info("Классификация тестовой выборки: %d/%d", index, len(samples))
    if not y_true:
        raise ValueError("Ни одно тестовое изображение не удалось обработать")

    # метрики качества
    accuracy = float(accuracy_score(y_true, y_pred))
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    logger.info("ТОЧНОСТЬ НА ТЕСТОВОЙ ВЫБОРКЕ: %.2f%% (%d верных из %d)", accuracy * 100, correct, len(y_true))

    report_labels = sorted(set(y_true) | set(y_pred))
    target_names = None
    if class_names and max(report_labels) < len(class_names):
        target_names = [class_names[label] for label in report_labels]
    report = classification_report(y_true, y_pred, labels=report_labels, target_names=target_names,
                                   digits=3, zero_division=0)
    logger.info("Отчёт по классам:\n%s", report)
    matrix = confusion_matrix(y_true, y_pred, labels=report_labels)
    logger.info("Матрица ошибок (строки — истинный класс, столбцы — предсказанный):\n%s", matrix)

    if getattr(args, "results_out", None):
        _save_results_csv(args.results_out, image_names, y_true, y_pred, class_names)
    if getattr(args, "report_out", None):
        _save_report_txt(args.report_out, accuracy, correct, len(y_true), report, matrix, report_labels, class_names)
    return accuracy


def _save_results_csv(path, image_names: Sequence[str], y_true: Sequence[int],
                      y_pred: Sequence[int], class_names: Sequence[str]) -> None:
    """Сохранить построчные результаты классификации в CSV (разделитель ';', кодировка для Excel)."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["изображение", "истинный класс", "предсказанный класс", "верно"])
        for name, true_label, pred_label in zip(image_names, y_true, y_pred):
            writer.writerow([
                name,
                _label_name(true_label, class_names),
                _label_name(pred_label, class_names),
                "да" if true_label == pred_label else "нет",
            ])
    logger.info("Результаты классификации сохранены: %s", report_path)


def _save_report_txt(path, accuracy: float, correct: int, total: int, report: str,
                     matrix: np.ndarray, labels: Sequence[int], class_names: Sequence[str]) -> None:
    """Сохранить текстовый отчёт: точность, отчёт по классам, матрица ошибок."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Точность классификации: {accuracy:.4f} ({correct} верных из {total})\n\n")
        file.write("Классы: " + ", ".join(_label_name(label, class_names) for label in labels) + "\n\n")
        file.write("Отчёт по классам:\n" + report + "\n\n")
        file.write("Матрица ошибок (строки — истинный класс, столбцы — предсказанный):\n")
        file.write(np.array2string(matrix))
        file.write("\n")
    logger.info("Отчёт сохранён: %s", report_path)



def run_predict(args) -> int:
    """Режим predict: загрузка модели, BoW-гистограмма входного изображения,
    классификация, сохранение графика гистограммы и совместной визуализации.
    Возвращает индекс предсказанного класса."""
    logger.info("=== РЕЖИМ PREDICT ===")
    bow = BagOfWords.load_model(args.model_path)
    logger.info("Параметры экстрактора берутся из файла модели (аргументы --extractor и подобные игнорируются)")

    image = read_image(args.image)
    height, width = image.shape[:2]
    logger.info("Изображение '%s': %dx%d", args.image, width, height)

    histogram = bow.histogram_from_image(image)
    predicted_label = bow.predict_histogram(histogram)
    predicted_name = _label_name(predicted_label, bow.class_names)
    logger.info("ПРЕДСКАЗАННЫЙ КЛАСС: %d (%s)", predicted_label, predicted_name)

    probabilities = bow.predict_proba_histogram(histogram)
    if probabilities is not None and bow.class_names and len(bow.class_names) == len(probabilities):
        top = np.argsort(probabilities)[::-1][:3]
        logger.info("Уверенность: %s",
                    "; ".join(f"{bow.class_names[i]} = {probabilities[i]:.2f}" for i in top))

    output_dir = ensure_dir(Path(args.output_dir))
    stem = Path(args.image).stem
    histogram_path = output_dir / f"{stem}_bow_histogram.png"
    joint_path = output_dir / f"{stem}_image_with_histogram.png"
    plot_histogram(
        histogram,
        title=f"BoW-гистограмма (экстрактор: {bow.extractor.name}) — предсказанный класс: {predicted_name}",
        save_path=str(histogram_path),
        show=args.show,
    )
    visualize_image_with_histogram(image, histogram, predicted_label=predicted_name,
                                   save_path=str(joint_path), show=args.show)
    logger.info("Результаты сохранены в каталог: %s", output_dir)
    return predicted_label
