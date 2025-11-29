import argparse
from bow_classifier import BOWClassifier
from nn_classifier import NNClassifier
from utils import load_dataset, evaluate_classifier, get_class_distribution


def main():
    parser = argparse.ArgumentParser(description='Классификация изображений достопримечательностей НН')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Путь к директории с данными (корневая директория с NNSUDataset и ExtDataset)')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Файл разбиения на train/test или директория с файлами разбиения')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], required=True,
                        help='Режим работы')
    parser.add_argument('--algorithm', choices=['bow', 'nn'], required=True,
                        help='Алгоритм классификации')
    parser.add_argument('--model_path', type=str, default='model.pkl',
                        help='Путь для сохранения/загрузки модели')

    # Параметры для BOW
    parser.add_argument('--vocab_size', type=int, default=100,
                        help='Размер словаря для BOW')
    parser.add_argument('--detector', type=str, default='SIFT',
                        choices=['SIFT', 'ORB', 'AKAZE'],
                        help='Детектор для BOW')

    # Параметры для нейросети
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')

    args = parser.parse_args()

    # Загрузка данных
    print("Загрузка данных...")
    train_images, train_labels, test_images, test_labels = load_dataset(args.data_path, args.split_file)

    if len(train_images) == 0 and args.mode in ['train', 'both']:
        print("Ошибка: Нет тренировочных данных для обучения!")
        return

    if len(test_images) == 0 and args.mode in ['test', 'both']:
        print("Ошибка: Нет тестовых данных для тестирования!")
        return

    # Вывод информации о данных
    if train_images:
        print(f"Распределение классов в тренировочной выборке: {get_class_distribution(train_labels)}")
    if test_images:
        print(f"Распределение классов в тестовой выборке: {get_class_distribution(test_labels)}")

    # Инициализация классификатора
    if args.algorithm == 'bow':
        classifier = BOWClassifier(vocab_size=args.vocab_size, detector=args.detector)
        model_ext = '.pkl'
    else:
        classifier = NNClassifier()
        model_ext = '.h5'

    # Корректировка пути модели
    if args.model_path == 'model.pkl':
        args.model_path = f'{args.algorithm}_model{model_ext}'

    # Обучение
    if args.mode in ['train', 'both']:
        print("Обучение классификатора...")
        classifier.train(train_images, train_labels, args.model_path)

    # Тестирование
    if args.mode in ['test', 'both']:
        print("Тестирование классификатора...")
        if args.mode == 'test':
            classifier.load(args.model_path)

        predictions = classifier.predict(test_images)
        accuracy, report = evaluate_classifier(test_labels, predictions)

        print(f"Точность: {accuracy:.4f}")
        print("Отчет классификации:")
        print(report)


if __name__ == "__main__":
    main()