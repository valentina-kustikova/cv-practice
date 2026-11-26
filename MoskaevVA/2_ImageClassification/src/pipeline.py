from __future__ import annotations
import csv
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from .bow import BagOfWords
from .data_loader import LandmarkDataLoader
from .extractors import FeatureExtractor
from .visualization import plot_histogram, visualize_image_with_histogram

LOG = logging.getLogger(__name__)


def _collect(loader, split, extractor):
    records, descriptors = [], []
    for i, record in enumerate(loader.load(split), 1):
        image = loader.read_image(record.path)
        if image is None:
            continue
        try:
            desc = extractor.detect_and_compute(image)
        except Exception as exc:
            LOG.warning("Ошибка извлечения признаков %s: %s", record.path, exc)
            continue
        records.append(record)
        descriptors.append(desc)
        if i % 100 == 0:
            LOG.info("%s: обработано %d изображений", split, i)
    LOG.info("%s: успешно обработано %d изображений", split, len(records))
    return records, descriptors


def run_train(
    data_dir,
    model_path,
    extractor_name="sift",
    clusters=128,
    classifier="svm",
    max_features=1000,
    device=None,
    layer=None,
):
    loader = LandmarkDataLoader(data_dir)
    extractor = (
        FeatureExtractor.create(extractor_name, max_features=max_features)
        if extractor_name in {"sift", "kaze"}
        else FeatureExtractor.create(extractor_name, device=device, layer=layer)
    )
    records, descriptors = _collect(loader, "trainval", extractor)
    if not records:
        raise RuntimeError("В trainval не удалось прочитать ни одного изображения.")
    bow = BagOfWords(clusters, classifier)
    bow.build_vocabulary(descriptors)
    X = np.vstack([bow.histogram(d) for d in descriptors])
    y = [r.label for r in records]
    bow.fit(X, y)
    bow.save_model(model_path, extractor.get_config())
    LOG.info("Обучение завершено: изображений=%d, классов=%d", len(records), len(set(y)))
    return bow


def run_test(data_dir, model_path, output_dir="outputs", device=None):
    loader = LandmarkDataLoader(data_dir)
    bow = BagOfWords.load_model(model_path)
    cfg = bow.extractor_config
    extractor_name = cfg.get("name", "sift")
    if extractor_name in {"sift", "kaze"}:
        extractor = FeatureExtractor.create(
            extractor_name, max_features=cfg.get("max_features", 1000)
        )
    else:
        extractor = FeatureExtractor.create(
            extractor_name, device=device or cfg.get("device"), layer=cfg.get("layer")
        )
    records, descriptors = _collect(loader, "test", extractor)
    if not records:
        raise RuntimeError("В test не удалось прочитать ни одного изображения.")
    y_true = [r.label for r in records]
    y_pred = [bow.predict(bow.histogram(d)) for d in descriptors]
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    LOG.info("Test accuracy: %.4f", accuracy)
    print(f"\nИзображений: {len(records)}\nAccuracy: {accuracy:.4f}\n\n{report}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true_label", "predicted_label"])
        for rec, true, pred in zip(records, y_true, y_pred):
            writer.writerow([str(rec.path), true, pred])
    with (out / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.6f}\n\n{report}\n")
    return accuracy


def run_predict(image_path, model_path, output_dir="outputs", device=None, show=False):
    bow = BagOfWords.load_model(model_path)
    cfg = bow.extractor_config
    name = cfg.get("name", "sift")
    if name in {"sift", "kaze"}:
        extractor = FeatureExtractor.create(name, max_features=cfg.get("max_features", 1000))
    else:
        extractor = FeatureExtractor.create(
            name, device=device or cfg.get("device"), layer=cfg.get("layer")
        )
    image = LandmarkDataLoader.read_image(image_path)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    desc = extractor.detect_and_compute(image)
    hist = bow.histogram(desc)
    prediction = bow.predict(hist)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    visualize_image_with_histogram(
        image,
        hist,
        out / "prediction.png",
        title=f"Предсказанный класс: {prediction}",
        show=show,
    )
    print(f"Предсказанный класс: {prediction}")
    print(f"Гистограмма сохранена: {out / 'prediction.png'}")
    return prediction
