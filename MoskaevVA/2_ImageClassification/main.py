from __future__ import annotations
import argparse
import logging
from src.pipeline import run_train, run_test, run_predict


def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description="Классификация достопримечательностей на основе Bag of Visual Words."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Обучить словарь и классификатор на trainval.")
    train.add_argument("--data", required=True, help="Корневая директория датасета.")
    train.add_argument("--model", default="outputs/model.joblib")
    train.add_argument(
        "--extractor",
        choices=["sift", "kaze", "resnet18", "convnext_tiny"],
        default="sift",
    )

    train.add_argument("--clusters", type=int, default=128)
    train.add_argument("--classifier", choices=["svm", "random_forest"], default="svm")
    train.add_argument("--max-features", type=int, default=1000)
    train.add_argument("--device", choices=["cpu", "cuda"], default=None)
    train.add_argument(
        "--layer",
        default=None,
        help="Имя слоя CNN (например layer3, layer4, features.12).",
    )

    test = sub.add_parser("test", help="Оценить сохраненную модель на test.")
    test.add_argument("--data", required=True)
    test.add_argument("--model", default="outputs/model.joblib")
    test.add_argument("--output", default="outputs")
    test.add_argument("--device", choices=["cpu", "cuda"], default=None)

    predict = sub.add_parser("predict", help="Классифицировать одно изображение.")
    predict.add_argument("--image", required=True)
    predict.add_argument("--model", default="outputs/model.joblib")
    predict.add_argument("--output", default="outputs")
    predict.add_argument("--device", choices=["cpu", "cuda"], default=None)
    predict.add_argument(
        "--show",
        action="store_true",
        help="Открыть окно с изображением и гистограммой.",
    )
    return parser


def main():
    parser = cli_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        if args.mode == "train":
            run_train(
                args.data,
                args.model,
                args.extractor,
                args.clusters,
                args.classifier,
                args.max_features,
                args.device,
                args.layer,
            )
        elif args.mode == "test":
            run_test(args.data, args.model, args.output, args.device)
        elif args.mode == "predict":
            run_predict(args.image, args.model, args.output, args.device, args.show)
    except Exception as exc:
        logging.exception("Выполнение завершилось с ошибкой: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
