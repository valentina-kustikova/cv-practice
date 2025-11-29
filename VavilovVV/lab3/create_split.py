import os
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger()

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def create_split(args):

    root_dir = Path(args.data_dir)
    if not root_dir.is_dir():
        log.error(f"Указанная директория не существует: {root_dir}")
        return

    all_images = []
    log.info(f"Сканирование {root_dir} (подпапки: {args.subfolders})...")

    for dataset_name in args.subfolders:
        dataset_path = root_dir / dataset_name
        if not dataset_path.is_dir():
            log.warning(f"Подпапка не найдена, пропуск: {dataset_path}")
            continue

        for f in dataset_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS:
                rel_path = f.relative_to(root_dir)
                all_images.append(str(rel_path.as_posix()))

    if not all_images:
        log.error(f"Изображения не найдены в {root_dir} / {args.subfolders}")
        return

    log.info(f"Найдено всего изображений: {len(all_images)}")

    train_files, test_files = train_test_split(
        all_images,
        test_size=args.test_size,
        random_state=42,
        shuffle=True
    )

    train_list_path = root_dir / args.train_file
    test_list_path = root_dir / args.test_file

    try:
        with open(train_list_path, "w") as f:
            f.write("\n".join(train_files))
        log.info(f"Train: {len(train_files)} изображений -> {train_list_path}")

        with open(test_list_path, "w") as f:
            f.write("\n".join(test_files))
        log.info(f"Test: {len(test_files)} изображений -> {test_list_path}")

        overlap = set(train_files) & set(test_files)
        if overlap:
            log.warning(f"Обнаружено пересечений: {len(overlap)}")
        else:
            log.info("Пересечений нет. Всё корректно!")

    except IOError as e:
        log.error(f"Ошибка записи файла: {e}")

def main():
    parser = argparse.ArgumentParser(description="Создание train/test списков изображений.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Корневая папка с данными (где лежат NNSUDataset и ExtDataset)"
    )
    parser.add_argument(
        "--subfolders",
        nargs='+',
        default=["NNSUDataset", "ExtDataset"],
        help="Список подпапок для сканирования"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля изображений для тестовой выборки"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.txt",
        help="Имя выходного файла для списка обучения (относительно data-dir)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="test.txt",
        help="Имя выходного файла для списка тестов (относительно data-dir)"
    )

    args = parser.parse_args()
    create_split(args)

if __name__ == "__main__":
    main()