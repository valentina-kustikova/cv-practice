"""Классификация изображений методом "мешок визуальных слов" (BoW).

Режимы запуска:
    python main.py train   --data-root DATA --extractor sift --model models/sift.joblib
    python main.py test    --data-root DATA --model models/sift.joblib --output results/sift.csv
    python main.py predict --model models/sift.joblib --image photo.jpg --save-dir results
"""

import argparse
import logging
import sys

from bow import CLASSIFIERS, DATA_LOADERS, FeatureExtractor
from bow.pipelines import run_predict, run_test, run_train

logger = logging.getLogger("bow")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Классификация изображений методом мешка визуальных слов")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, help="дополнительно писать лог в файл")
    sub = parser.add_subparsers(dest="mode", required=True)

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data-root", required=True, help="корневой каталог набора данных")
    data.add_argument("--dataset", default="intel", choices=sorted(DATA_LOADERS), help="формат набора данных")
    data.add_argument("--limit-per-class", type=int, default=None,
                      help="ограничить число изображений на класс (для отладки)")

    device = argparse.ArgumentParser(add_help=False)
    device.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="устройство PyTorch для CNN (по умолчанию auto: CUDA при наличии, иначе CPU)")

    train = sub.add_parser("train", parents=[data, device], help="обучение модели")
    train.add_argument("--extractor", required=True, choices=FeatureExtractor.available())
    train.add_argument("--model", required=True, help="путь для сохранения модели (.joblib)")
    train.add_argument("--vocab-size", type=int, default=200, help="число визуальных слов K")
    train.add_argument("--classifier", default="svm", choices=sorted(CLASSIFIERS))
    train.add_argument("--vocab-images", type=int, default=2000,
                       help="число изображений, по дескрипторам которых строится словарь")
    train.add_argument("--max-descriptors", type=int, default=200_000,
                       help="максимум дескрипторов для K-средних")
    train.add_argument("--pca-dim", type=int, default=None,
                       help="PCA перед K-средних (0 – отключить; по умолчанию 128 для CNN, нет для классических)")
    train.add_argument("--tfidf", action="store_true", help="взвешивать слова по TF-IDF")
    train.add_argument("--seed", type=int, default=42)
    # Параметры классических экстракторов
    train.add_argument("--image-size", type=int, default=None, help="SIFT/ORB: размер большей стороны (300)")
    train.add_argument("--nfeatures", type=int, default=None, help="SIFT/ORB: максимум ключевых точек (1000)")
    # Параметры нейросетевых экстракторов
    train.add_argument("--layer", default=None, help="CNN: слой для извлечения признаков (layer4, features.8, ...)")
    train.add_argument("--input-size", type=int, default=None, help="CNN: размер входа сети (320)")

    test = sub.add_parser("test", parents=[data, device], help="тестирование модели")
    test.add_argument("--model", required=True)
    test.add_argument("--output", default=None, help="CSV-файл с результатами классификации")

    predict = sub.add_parser("predict", parents=[device], help="классификация одного изображения")
    predict.add_argument("--model", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--save-dir", default=None, help="каталог для сохранения графиков")
    predict.add_argument("--no-show", action="store_true", help="не открывать окна с графиками")
    predict.add_argument("--draw-keypoints", action="store_true", help="отрисовать ключевые точки")

    return parser.parse_args(argv)


def setup_logging(level, log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=handlers)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_level, args.log_file)
    modes = {"train": run_train, "test": run_test, "predict": run_predict}
    try:
        logger.info("Режим: %s", args.mode)
        modes[args.mode](args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("%s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Прервано пользователем")
        return 130
    except Exception:
        logger.exception("Непредвиденная ошибка")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
