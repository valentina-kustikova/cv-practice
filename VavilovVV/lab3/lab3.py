import os
import argparse
import logging
from pathlib import Path

# Импортируем наши новые классы
from bow_classifier import BowClassifier
from cnn_classifier import CnnClassifier

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description="Классификатор изображений (BoW/NN)")

    parser.add_argument("--data-dir", type=str, required=True,
                        help="Корневая папка с данными (где лежат train.txt/test.txt и папки с фото)")
    parser.add_argument("--train-file", type=str, required=True,
                        help="Имя файла train.txt (относительно data-dir)")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Имя файла test.txt (относительно data-dir)")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Папка для сохранения моделей")

    # --- Режимы работы ---
    parser.add_argument("--mode", type=str, required=True,
                        help="Режимы работы, через запятую (e.g., 'train', 'test', 'train,test', 'visualize')")
    parser.add_argument("--algorithm", type=str, required=True, choices=['bow', 'nn'],
                        help="Алгоритм для использования ('bow' или 'nn')")

    # --- Общие параметры ---
    parser.add_argument("--classes", nargs='+',
                        default=['01_NizhnyNovgorodKremlin', '04_ArkhangelskCathedral', '08_PalaceOfLabor'],
                        help="Список имен классов")

    # --- BoW параметры ---
    parser.add_argument("--k", type=int, default=500,
                        help="[BoW] Размер словаря (кол-во кластеров KMeans)")
    parser.add_argument("--detector", type=str, default="sift", choices=['sift', 'orb'],
                        help="[BoW] Детектор ключевых точек")
    parser.add_argument("--svm-kernel", type=str, default="linear",
                        help="[BoW] Ядро SVM (linear, rbf, poly)")
    parser.add_argument("--svm-c", type=float, default=1.0,
                        help="[BoW] Параметр регуляризации SVM")

    # --- NN параметры ---
    parser.add_argument("--epochs", type=int, default=20,
                        help="[NN] Количество эпох обучения")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="[NN] Скорость обучения (learning rate)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[NN] Размер батча")
    parser.add_argument("--img-size", type=int, default=224,
                        help="[NN] Размер изображения (img_size x img_size)")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="[NN] Не использовать предобученные веса ImageNet")

    # Устанавливаем default=True для pretrained
    parser.set_defaults(pretrained=True)

    parser.add_argument("--num-workers", type=int, default=os.cpu_count() // 2,
                        help="[NN] Количество потоков для загрузки данных")

    args = parser.parse_args()

    # Преобразуем пути к файлам в абсолютные
    args.train_file = str(Path(args.data_dir) / args.train_file)
    args.test_file = str(Path(args.data_dir) / args.test_file)

    # --- Выбор и запуск классификатора ---

    classifier = None
    if args.algorithm == 'bow':
        classifier = BowClassifier(args)
    elif args.algorithm == 'nn':
        classifier = CnnClassifier(args)
    else:
        log.error(f"Неизвестный алгоритм: {args.algorithm}")
        return

    modes = args.mode.split(',')

    try:
        if 'train' in modes:
            classifier.train()

        if 'test' in modes:
            classifier.test()

        if 'visualize' in modes:
            if isinstance(classifier, BowClassifier):
                classifier.visualize_keypoints()
            else:
                log.warning("Визуализация доступна только для 'bow' алгоритма.")

    except FileNotFoundError as e:
        log.error(f"Ошибка: {e}")
        log.error("Файл модели не найден. Запустите 'train' режим для создания модели.")
    except Exception as e:
        log.exception(f"Произошла непредвиденная ошибка: {e}")

if __name__ == "__main__":
    main()
