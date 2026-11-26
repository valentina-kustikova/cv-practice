from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .bow import BagOfWords
from .data_loader import ImageRecord, LandmarkDataLoader
from .extractors import FeatureExtractor, LAYER_EXTRACTORS, NEURAL_EXTRACTORS
from .utils import read_image
from .visualization import plot_histogram, visualize_image_with_histogram

logger = logging.getLogger(__name__)


def _load_descriptors(
    records: list[ImageRecord],
    extractor: FeatureExtractor,
    log_every: int = 50,
) -> tuple[list[ImageRecord], list[np.ndarray]]:
    """Extract descriptors, skipping files that cannot be decoded.

    The returned records and descriptor sets always have matching lengths and
    order. This is important because labels are taken from the returned records.
    Errors raised by the feature extractor are not suppressed.
    """
    valid_records: list[ImageRecord] = []
    descriptor_sets: list[np.ndarray] = []
    skipped = 0

    for index, record in enumerate(records, start=1):
        try:
            image = read_image(record.path)
        except (OSError, ValueError) as exc:
            skipped += 1
            logger.warning(
                "Skipping unreadable image %d/%d: %s (%s)",
                index,
                len(records),
                record.path,
                exc,
            )
        else:
            descriptors = extractor.detect_and_compute(image)
            valid_records.append(record)
            descriptor_sets.append(descriptors)

        if index % log_every == 0 or index == len(records):
            logger.info(
                "Processed images: %d/%d; valid=%d; skipped=%d",
                index,
                len(records),
                len(valid_records),
                skipped,
            )

    if not valid_records:
        raise RuntimeError(
            "No readable images remain after checking the dataset. "
            "Verify the image files and dataset path."
        )

    if skipped:
        logger.warning(
            "Skipped %d unreadable image(s) out of %d. "
            "The remaining images will be used.",
            skipped,
            len(records),
        )

    return valid_records, descriptor_sets


def _create_loader(args) -> LandmarkDataLoader:
    return LandmarkDataLoader(
        root=args.dataset,
        include_external=not args.no_external,
        include_other=not args.exclude_other,
    )


def run_train(args) -> None:
    extractor_kwargs = {}
    if args.extractor in NEURAL_EXTRACTORS:
        extractor_kwargs.update(device=args.device, pretrained=not args.no_pretrained)
    if args.window_size is not None:
        if not extractor_kwargs:
            raise ValueError("--window-size requires a CNN extractor")
        if args.window_size < 1:
            raise ValueError("--window-size must be a positive integer")
        extractor_kwargs["window_size"] = args.window_size
    if args.layer is not None:
        if args.extractor not in LAYER_EXTRACTORS:
            raise ValueError(f"--layer requires one of {', '.join(LAYER_EXTRACTORS)}")
        extractor_kwargs["layer"] = args.layer

    loader = _create_loader(args)
    records = loader.load("trainval")
    logger.info(
        "Training split discovered: %d images; external=%s; include 00_Other=%s",
        len(records),
        not args.no_external,
        not args.exclude_other,
    )

    extractor = FeatureExtractor.create(args.extractor, **extractor_kwargs)
    if args.extractor in LAYER_EXTRACTORS:
        logger.info(
            "%s on %s: window=%d, layer=%d (%s)",
            args.extractor, extractor.device, extractor.window_size, extractor.layer, extractor.layer_names[extractor.layer],
        )
    records, descriptor_sets = _load_descriptors(records, extractor)
    labels = [record.label for record in records]
    logger.info(
        "Training images available: %d, %d classes",
        len(records),
        len(set(labels)),
    )

    bow = BagOfWords(
        extractor=extractor,
        extractor_name=args.extractor,
        n_clusters=args.clusters,
        classifier_name=args.classifier,
        random_state=args.seed,
    )
    bow.fit(descriptor_sets, labels, max_descriptors=args.max_descriptors)
    saved_path = bow.save_model(args.model_out)
    logger.info("Model saved to %s", saved_path)
    logger.info("Vocabulary size: %d visual words", bow.n_clusters)


def run_test(args) -> None:
    model = BagOfWords.load_model(args.model, device=args.device)
    loader = _create_loader(args)
    records = loader.load("test")
    logger.info(
        "Testing split discovered: %d images; extractor=%s",
        len(records),
        model.extractor_name,
    )

    # Test images are processed one at a time. This avoids retaining every
    # CNN descriptor matrix in RAM (especially important for ResNet-50, whose
    # 49 descriptors have 2048 values each).
    valid_records: list[ImageRecord] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    skipped = 0

    for index, record in enumerate(records, start=1):
        try:
            image = read_image(record.path)
        except (OSError, ValueError) as exc:
            skipped += 1
            logger.warning(
                "Skipping unreadable test image %d/%d: %s (%s)",
                index,
                len(records),
                record.path,
                exc,
            )
        else:
            histogram = model.histogram(image)
            prediction = model.classifier.predict(histogram.reshape(1, -1))[0]
            valid_records.append(record)
            true_labels.append(record.label)
            predicted_labels.append(str(prediction))
            del image, histogram

        if index % 50 == 0 or index == len(records):
            logger.info(
                "Tested images: %d/%d; valid=%d; skipped=%d",
                index,
                len(records),
                len(valid_records),
                skipped,
            )

    if not valid_records:
        raise RuntimeError("No readable test images remain; cannot calculate metrics")

    if skipped:
        logger.warning("Skipped %d unreadable test image(s) out of %d", skipped, len(records))

    records = valid_records
    y_true = np.asarray(true_labels, dtype=str)
    y_pred = np.asarray(predicted_labels, dtype=str)
    logger.info(
        "Testing images available: %d, %d classes",
        len(records),
        len(set(y_true)),
    )
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info("Test accuracy: %.4f (%.2f%%)", accuracy, accuracy * 100)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print(report)

    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "true_label", "predicted_label", "correct"])
            for record, true_label, predicted_label in zip(records, y_true, y_pred):
                writer.writerow([str(record.path), true_label, predicted_label, true_label == predicted_label])
        logger.info("Per-image predictions saved to %s", output_path)

    if args.save_confusion_matrix:
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        labels = sorted(set(y_true) | set(y_pred))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), max(7, len(labels) * 0.4)))
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False
        )
        fig.tight_layout()
        output = Path(args.save_confusion_matrix)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", output)


def run_predict(args) -> None:
    model = BagOfWords.load_model(args.model, device=args.device)
    image = read_image(args.image)
    predicted_label, histogram = model.predict_with_histogram(image)
    print(f"Predicted class: {predicted_label}")
    logger.info("Predicted class: %s", predicted_label)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_histogram(
        histogram,
        save_path=output_dir / "bow_histogram.png",
        show=args.show,
        title=f"BoW histogram — {predicted_label}",
    )
    visualize_image_with_histogram(
        image,
        histogram,
        save_path=output_dir / "image_and_histogram.png",
        show=args.show,
        title=f"Predicted class: {predicted_label}",
    )
    import matplotlib.pyplot as plt

    plt.close("all")
    logger.info("Visualization saved to %s", output_dir.resolve())
