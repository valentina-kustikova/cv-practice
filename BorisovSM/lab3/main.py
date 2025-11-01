import argparse
from data_reader import get_train_test
from bow import train_bow, predict_bow, save_bow, load_bow, save_keypoints_for_items
from cnn import train_cnn, eval_cnn


def cli_argument_parser():
    parser = argparse.ArgumentParser(description="Classification")

    parser.add_argument("--data_dir", required=True, type=str,
                        help="Корневая директория с ExtDataset/ и NNSUDataset/")
    parser.add_argument("--train_list", required=True, type=str,
                        help="Файл со списком путей тренировочной выборки")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                        help="train: обучение; test: тест по загруженной модели; both: обучение+тест")
    parser.add_argument("--algo", choices=["bow", "cnn"], default="bow",
                        help="Какой алгоритм использовать: мешок слов (bow) или ResNet50 (cnn)")

    parser.add_argument("--k", type=int, default=300, help="Размер словаря BoW (KMeans)")
    parser.add_argument("--C", type=float, default=1.0, help="Параметр C для LinearSVC")
    parser.add_argument("--max_kp", type=int, default=1000,
                        help="Макс. число ключевых точек (SIFT)")

    parser.add_argument("--vizualize_dir", type=str, help="Путь к сохранению изображений с ключевыми точками SIFT")
    parser.add_argument("--viz_per_class", type=int, default=2,
                        help="Сколько изображений на класс визуализировать (SIFT)")

    parser.add_argument("--model_in", type=str, help="Путь к сохранённой модели (pickle)")
    parser.add_argument("--model_out", type=str, help="Куда сохранить обученную модель (pickle)")
    
    parser.add_argument("--val_size", type=float, default=0.2,
                    help="Доля валидации, выделяемая из train (CNN)")
    parser.add_argument("--epochs", type=int, default=8, help="Эпохи обучения CNN")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size для CNN")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate для CNN")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Устройство для CNN: cuda или cpu")
    parser.add_argument("--weights_in", type=str, help="Файл весов CNN (.pth) для режима test")
    parser.add_argument("--weights_out", type=str, default="resnet50.pth",
                        help="Куда сохранить лучшие веса CNN (.pth)")

    return parser.parse_args()


def main():
    args = cli_argument_parser()

    train_items, test_items = get_train_test(args.data_dir, args.train_list)
    print(f"Размер тренировочной выборки: {len(train_items)}")
    print(f"Размер тестовой выборки: {len(test_items)}")

    if args.algo == "bow":

        if args.vizualize_dir:
            save_keypoints_for_items(train_items, out_dir=args.vizualize_dir, per_class=args.viz_per_class,
                max_kp=args.max_kp)

        if args.mode == "train":
            model = train_bow(train_items, k=args.k, C=args.C, max_kp=args.max_kp)
            print("Модель обучена")
            if args.model_out:
                save_bow(model, args.model_out)
                print(f"Модель сохранена: {args.model_out}")

        elif args.mode == "test":
            if not args.model_in:
                raise ValueError("Для режима test укажите --model_in (путь к сохранённой модели).")
            model = load_bow(args.model_in)
            print(f"Модель загружена: {args.model_in}")
            _, _, _, report = predict_bow(model, test_items, max_kp=args.max_kp)
            print(f"Accuracy: {report['accuracy']:.4f}")
            print("Classification report:\n", report["classification_report"])
            print("Confusion matrix:\n", report["confusion_matrix"])

        else:
            model = train_bow(train_items, k=args.k, C=args.C, max_kp=args.max_kp)
            print("Модель обучена.")
            if args.model_out:
                save_bow(model, args.model_out)
                print(f"Модель сохранена: {args.model_out}")
            _, _, _, report = predict_bow(model, test_items)
            print(f"Accuracy: {report['accuracy']:.4f}")
            print("Classification report:\n", report["classification_report"])
            print("Confusion matrix:\n", report["confusion_matrix"])

    elif args.algo == "cnn":
        dev = args.device if args.device in ("cpu", "cuda") else "cuda"
        if args.mode == "train":
            train_cnn(
                train_items,
                epochs=args.epochs, lr=args.lr, bs=args.batch_size,
                device=dev, weights_out=args.weights_out
            )

        elif args.mode == "test":
            if not args.weights_in:
                raise ValueError("Для режима test укажите --weights_in (путь к .pth весам CNN)")
            eval_cnn(test_items, device=dev, weights_path=args.weights_in)

        else:
            train_cnn(
                train_items,
                epochs=args.epochs, lr=args.lr, bs=args.batch_size,
                device=dev, weights_out=args.weights_out
            )
            eval_cnn(test_items, device=dev, weights_path=args.weights_out)


if __name__ == "__main__":
    main()
