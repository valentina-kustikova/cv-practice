#!/usr/bin/env python3
"""Практическая работа №2. Классификация изображений с использованием библиотеки OpenCV.

Метод «мешок визуальных слов» (Bag-of-Words) на основе классических дескрипторов
(SIFT, ORB) и признаков нейросетей (ResNet-18, MobileNetV2), извлекаемых с
заданного слоя, с различными методами машинного обучения (SVM, RF, kNN).

Режимы запуска:
    train   — построение словаря визуальных слов, BoW-гистограмм, обучение и сохранение классификатора;
    test    — оценка точности классификации на тестовой выборке;
    predict — классификация одного изображения с визуализацией BoW-гистограммы.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.pipelines import run_predict, run_test, run_train
from src.utils import set_seed, setup_logging

logger = logging.getLogger("src.main")

EXAMPLES = """примеры запуска:
  python main.py train --data-root ./Dataset --extractor orb --vocab-size 100 --classifier rf --max-per-class 10
  python main.py train --dataset ./Dataset --extractor resnet18 --vocab-size 500
  python main.py test --data-root ./Dataset --results-out ./outputs/results.csv --report-out ./outputs/report.txt
  python main.py predict --image ./test_images/photo.jpg --output-dir ./outputs
"""


def build_parser() -> argparse.ArgumentParser:
    """Построить парсер аргументов командной строки: три подкоманды-режима train/test/predict."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Классификация изображений методом «мешка визуальных слов» (Bag-of-Words): "
                    "классические дескрипторы OpenCV (SIFT, ORB) и признаки CNN (PyTorch).",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, metavar="{train,test,predict}")

    # --- общие аргументы всех режимов ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--extractor", choices=("sift", "orb", "resnet18", "mobilenetv2"),
                        default="sift",
                        help="метод извлечения признаков (в режимах test/predict параметры берутся из файла модели)")
    common.add_argument("--dataset", default="folder",
                        help="тип источника данных: 'folder' — набор в виде дерева папок trainval/test "
                             "(по умолчанию); вместо имени можно передать прямой путь к каталогу набора, "
                             "например --dataset ./Dataset — эквивалентно --data-root ./Dataset")
    common.add_argument("--data-root", default="./data",
                        help="корень набора данных: каталог с подкаталогами trainval/ и test/ "
                             "(классы — каталоги, непосредственно содержащие изображения; "
                             "вложенность поднаборов любой глубины допустима)")
    common.add_argument("--model-path", default="./models/bow_model.pkl",
                        help="путь к файлу модели (словарь + классификатор)")
    common.add_argument("--max-per-class", type=int, default=0,
                        help="использовать не более N изображений на класс (0 — все; удобно для быстрых прогонов)")
    common.add_argument("--min-side", type=int, default=0,
                        help="увеличивать изображение, если меньшая сторона меньше этого значения, "
                             "перед классическим детектором (0 — не увеличивать; полезно для мелких изображений)")
    common.add_argument("--cnn-size", type=int, default=224,
                        help="размер входного изображения нейросети (для экстракторов resnet18/mobilenetv2)")
    common.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="устройство для нейросетевого экстрактора")
    common.add_argument("--no-pretrained", action="store_true",
                        help="не загружать предобученные ImageNet-веса нейросети (работа без интернета)")
    common.add_argument("--seed", type=int, default=42, help="зерно генератора случайных чисел")
    common.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
                        help="уровень логирования")

    # --- режим train ---
    parser_train = subparsers.add_parser("train", parents=[common],
                                         help="обучение модели на тренировочной выборке")
    parser_train.add_argument("--vocab-size", type=int, default=200,
                              help="размер словаря визуальных слов (число кластеров K-средних)")
    parser_train.add_argument("--classifier", choices=("svm", "rf", "knn"), default="svm",
                              help="метод машинного обучения, обучаемый на BoW-гистограммах")
    parser_train.add_argument("--hist-norm", choices=("l1", "l2", "none"), default="l1",
                              help="способ нормировки BoW-гистограмм")
    parser_train.add_argument("--vocab-sample-size", type=int, default=150000,
                              help="максимальное число дескрипторов для построения словаря (0 — использовать все); "
                                   "ускоряет K-средних на больших наборах")
    parser_train.add_argument("--max-descriptors-per-image", type=int, default=0,
                              help="ограничение числа дескрипторов с одного изображения (0 — без ограничения)")

    # --- режим test ---
    parser_test = subparsers.add_parser("test", parents=[common],
                                        help="оценка точности на тестовой выборке")
    parser_test.add_argument("--results-out", default=None,
                             help="сохранить CSV с результатами по каждому тестовому изображению")
    parser_test.add_argument("--report-out", default=None,
                             help="сохранить текстовый отчёт: точность, метрики по классам, матрица ошибок")

    # --- режим predict ---
    parser_predict = subparsers.add_parser("predict", parents=[common],
                                           help="классификация одного изображения")
    parser_predict.add_argument("--image", required=True, help="путь к классифицируемому изображению")
    parser_predict.add_argument("--output-dir", default="./outputs",
                                help="каталог для сохранения графиков BoW-гистограммы")
    parser_predict.add_argument("--show", action="store_true",
                                help="показать окна matplotlib в дополнение к сохранению в файлы")
    return parser


DATASET_TYPES = ("folder",)
# источники данных, удалённые из проекта (для понятного сообщения вместо «каталог не найден»)
REMOVED_DATASET_TYPES = {
    "cifar10": "Набор CIFAR-10 удалён из проекта: используйте набор данных в виде дерева папок "
               "trainval/test (--data-root ./Dataset).",
}


def resolve_dataset_args(args) -> None:
    """Нормализовать источник данных после разбора аргументов.

    --dataset принимает либо имя типа ('folder'), либо прямой путь к каталогу
    набора: в последнем случае путь подставляется в --data-root, а тип источника
    становится 'folder'. Обе формы записи приводятся к одному виду, поэтому
    пайплайны всегда работают с args.dataset == 'folder' и args.data_root.
    """
    if args.dataset in DATASET_TYPES:
        return
    if args.dataset in REMOVED_DATASET_TYPES:
        raise ValueError(REMOVED_DATASET_TYPES[args.dataset])
    path_value = args.dataset
    args.dataset = "folder"
    args.data_root = path_value
    logger.info("--dataset '%s' распознан как путь к каталогу набора (эквивалентно --data-root '%s')",
                path_value, path_value)


def main(argv=None) -> int:
    """Точка входа: разбор аргументов командной строки -> выбор режима -> выполнение
    соответствующего пайплайна с обработкой ошибок и логированием.

    Возвращает код возврата процесса: 0 — успех, 1 — ошибка, 130 — прервано пользователем.
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    set_seed(args.seed)
    try:
        resolve_dataset_args(args)
        logger.info("Старт: режим='%s', экстрактор='%s', набор данных='%s', данные='%s'",
                    args.mode, args.extractor, args.dataset, args.data_root)
        if args.mode == "train":
            run_train(args)
        elif args.mode == "test":
            run_test(args)
        else:
            run_predict(args)
    except KeyboardInterrupt:
        logger.warning("Выполнение прервано пользователем")
        return 130
    except Exception as exc:  # единая точка обработки ошибок приложения
        logger.exception("Критическая ошибка: %s", exc)
        logger.error("Выполнение завершено с ошибкой. Проверьте параметры запуска и структуру данных.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
