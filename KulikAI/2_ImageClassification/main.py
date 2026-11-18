from __future__ import annotations

import argparse
import logging
import sys

from image_classification.pipelines import run_predict, run_test, run_train


def cli_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Image classification with local features, CNN descriptors and Bag of Visual Words."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train a BoW vocabulary and image classifier")
    train_parser.add_argument("--dataset", required=True, help="Dataset root containing trainval/ and test/")
    train_parser.add_argument("--model-out", default="models/bow_model.joblib", help="Output model file")
    train_parser.add_argument(
        "--extractor",
        choices=("sift", "kaze", "resnet18", "resnet50", "mobilenet_v3_small"),
        default="sift",
        help="Descriptor extractor (default: sift)",
    )
    train_parser.add_argument(
        "--classifier",
        choices=("svm", "random_forest", "knn"),
        default="svm",
        help="Classifier trained on BoW histograms (default: svm)",
    )
    train_parser.add_argument("--clusters", type=int, default=128, help="Number of visual words")
    train_parser.add_argument(
        "--max-descriptors",
        type=int,
        default=100_000,
        help="Maximum descriptors used to fit K-means; 0 means no limit",
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--device", default="auto", help="CNN device: auto, cpu, cuda, cuda:0, ...")
    train_parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet weights (not recommended for feature quality)",
    )
    train_parser.add_argument("--no-external", action="store_true", help="Ignore ExtDataset")
    train_parser.add_argument("--exclude-other", action="store_true", help="Exclude the 00_Other class")

    test_parser = subparsers.add_parser("test", help="Evaluate a saved model on the test split")
    test_parser.add_argument("--dataset", required=True, help="Dataset root containing trainval/ and test/")
    test_parser.add_argument("--model", required=True, help="Path to a model created by train")
    test_parser.add_argument("--device", default="auto", help="CNN device: auto, cpu, cuda, cuda:0, ...")
    test_parser.add_argument("--no-external", action="store_true", help="Ignore ExtDataset")
    test_parser.add_argument("--exclude-other", action="store_true", help="Exclude the 00_Other class")
    test_parser.add_argument("--save-results", help="Optional CSV file with per-image predictions")
    test_parser.add_argument("--save-confusion-matrix", help="Optional PNG path for the confusion matrix")

    predict_parser = subparsers.add_parser("predict", help="Classify one image and visualize its BoW histogram")
    predict_parser.add_argument("--model", required=True, help="Path to a model created by train")
    predict_parser.add_argument("--image", required=True, help="Input image path")
    predict_parser.add_argument("--device", default="auto", help="CNN device: auto, cpu, cuda, cuda:0, ...")
    predict_parser.add_argument("--output-dir", default="outputs/predict", help="Directory for visualizations")
    predict_parser.add_argument("--show", action="store_true", help="Display figures interactively")

    return parser


def main() -> int:
    parser = cli_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.mode == "train":
            run_train(args)
        elif args.mode == "test":
            run_test(args)
        elif args.mode == "predict":
            run_predict(args)
        else:  
            parser.error(f"Unknown mode: {args.mode}")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        logging.exception("Application failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
