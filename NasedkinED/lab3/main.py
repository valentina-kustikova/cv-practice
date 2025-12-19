import argparse

from src.bag_of_words import BagOfWordsClassifier
from src.neural_network import TransferLearningClassifier
from src.utils import load_data_paths, print_metrics


def main(args):
    """Основная функция приложения."""

    # --- 1. Загрузка набора данных ---
    print(f"Загрузка данных из директории: {args.data_dir}")
    train_paths, train_labels, test_paths, test_labels, class_to_label = load_data_paths(
        args.data_dir,
        args.train_file
    )

    print(f"  Классов найдено: {len(class_to_label)}")
    print(f"  Тренировочная выборка: {len(train_paths)} изображений")
    print(f"  Тестовая выборка: {len(test_paths)} изображений")

    if not train_paths or not test_paths:
        print("Ошибка: Недостаточно данных для тренировки или тестирования. Проверьте пути и train.txt.")
        return

    # --- 2. Инициализация и настройка классификатора ---
    y_pred = []

    if args.algorithm == 'bag_of_words':
        classifier = BagOfWordsClassifier(
            n_clusters=args.bow_clusters,
            descriptor_type=args.bow_descriptor
        )

        if args.mode in ['train', 'full']:
            classifier.train(train_paths, train_labels, visualize=args.visualize_bow)

        if args.mode in ['test', 'full']:
            print("\nТестирование Bag-of-Words классификатора...")
            y_pred = classifier.predict(test_paths)

    elif args.algorithm == 'neural_network':
        num_classes = len(class_to_label)
        classifier = TransferLearningClassifier(num_classes)

        if args.mode in ['train', 'full']:
            classifier.train(
                train_paths,
                train_labels,
                num_epochs=args.nn_epochs,
                batch_size=args.nn_batch_size
            )

        if args.mode in ['test', 'full']:
            print("\nТестирование нейросетевого классификатора...")
            y_pred = classifier.predict(test_paths, batch_size=args.nn_batch_size)

    else:
        print("Неизвестный алгоритм.")
        return

    # --- 3. Вывод качества решения ---
    if args.mode in ['test', 'full'] and y_pred:
        print("\n\n=== РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ ===")
        test_labels_matched = test_labels[:len(y_pred)]
        print_metrics(test_labels_matched, y_pred, class_to_label)
        print("================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Классификация изображений достопримечательностей Нижнего Новгорода.")

    # Обязательные параметры
    parser.add_argument('data_dir', type=str, help="Путь до директории с данными (data/).")
    parser.add_argument('train_file', type=str, default='train.txt',
                        help="Имя файла разбиения на тренировочную выборку (e.g., train.txt).")
    parser.add_argument('mode', choices=['train', 'test', 'full'],
                        help="Режим работы: 'train', 'test' или 'full' (обучение и тестирование).")
    parser.add_argument('algorithm', choices=['bag_of_words', 'neural_network'],
                        help="Алгоритм работы: 'bag_of_words' или 'neural_network'.")

    # Параметры Bag of Words
    parser.add_argument('--bow_clusters', type=int, default=100, help="Количество кластеров (визуальных слов) для BoW.")
    parser.add_argument('--bow_descriptor', type=str, choices=['SIFT', 'ORB'], default='SIFT',
                        help="Тип дескриптора для BoW.")
    parser.add_argument('--visualize_bow', action='store_true',
                        help="Флаг для визуализации ключевых точек в режиме BoW.")

    # Параметры Нейронной сети
    parser.add_argument('--nn_epochs', type=int, default=10, help="Количество эпох обучения для Нейронной сети.")
    parser.add_argument('--nn_batch_size', type=int, default=8, help="Размер батча для Нейронной сети.")

    args = parser.parse_args()
    main(args)
