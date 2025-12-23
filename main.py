import argparse
import os
import sys
from pathlib import Path
from data_loader import DataLoader
from bag_of_words import BagOfWords
from neural_network import NeuralNetworkClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Классификация изображений достопримечательностей Нижнего Новгорода'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Путь до директории с данными'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='Путь к файлу со списком тренировочных изображений'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Режим работы: train (обучение), test (тестирование), both (обучение и тестирование)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['bow', 'nn', 'both'],
        default='both',
        help='Алгоритм: bow (мешок слов), nn (нейронная сеть), both (оба алгоритма)'
    )
    
    parser.add_argument(
        '--bow_vocab_size',
        type=int,
        default=200,
        help='Размер словаря для Bag of Words (по умолчанию: 200, рекомендуется 200-500 для лучшей точности)'
    )
    
    parser.add_argument(
        '--bow_detector',
        type=str,
        choices=['SIFT', 'ORB', 'AKAZE'],
        default='SIFT',
        help='Тип детектора для Bag of Words (по умолчанию: SIFT)'
    )
    
    parser.add_argument(
        '--bow_model_path',
        type=str,
        default='bow_model.pkl',
        help='Путь для сохранения/загрузки модели Bag of Words'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Создавать визуализации ключевых точек (для Bag of Words)'
    )
    
    parser.add_argument(
        '--nn_base_model',
        type=str,
        choices=['ResNet50', 'VGG16', 'MobileNetV2'],
        default='ResNet50',
        help='Базовая модель для transfer learning (по умолчанию: ResNet50)'
    )
    
    parser.add_argument(
        '--nn_epochs',
        type=int,
        default=20,
        help='Количество эпох обучения нейронной сети (по умолчанию: 20)'
    )
    
    parser.add_argument(
        '--nn_batch_size',
        type=int,
        default=16,
        help='Размер батча для нейронной сети (по умолчанию: 16)'
    )
    
    parser.add_argument(
        '--nn_learning_rate',
        type=float,
        default=0.0001,
        help='Скорость обучения нейронной сети (по умолчанию: 0.0001)'
    )
    
    parser.add_argument(
        '--nn_model_path',
        type=str,
        default='nn_model.h5',
        help='Путь для сохранения/загрузки модели нейронной сети'
    )
    
    return parser.parse_args()


def train_bow(data_loader, args):
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ BAG OF WORDS")
    print("="*60)
    
    print("Загрузка тренировочных данных...")
    train_images, train_labels = data_loader.load_train_images()
    print(f"Загружено {len(train_images)} тренировочных изображений")
    
    bow = BagOfWords(
        vocab_size=args.bow_vocab_size,
        detector_type=args.bow_detector,
        descriptor_type=args.bow_detector,
        use_tfidf=True
    )
    
    bow.train(train_images, train_labels, visualize=args.visualize)
    
    bow.save(args.bow_model_path)
    
    return bow


def test_bow(data_loader, args):
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ BAG OF WORDS")
    print("="*60)
    
    if not os.path.exists(args.bow_model_path):
        print(f"Ошибка: Модель {args.bow_model_path} не найдена!")
        return None
    
    bow = BagOfWords(
        vocab_size=args.bow_vocab_size,
        detector_type=args.bow_detector,
        descriptor_type=args.bow_detector,
        use_tfidf=True
    )
    bow.load(args.bow_model_path)
    
    print("Загрузка тестовых данных...")
    test_images, test_labels, test_paths = data_loader.load_test_images()
    print(f"Загружено {len(test_images)} тестовых изображений")
    
    if len(test_images) == 0:
        print("Тестовые данные не найдены!")
        return None
    
    accuracy = bow.evaluate(test_images, test_labels)
    print(f"\nТочность классификации (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    predictions = bow.predict(test_images)
    print("\nДетальная статистика по классам:")
    for i in range(data_loader.get_num_classes()):
        class_name = data_loader.get_class_name(i)
        class_indices = [j for j, label in enumerate(test_labels) if label == i]
        if len(class_indices) > 0:
            class_predictions = [predictions[j] for j in class_indices]
            class_accuracy = sum(1 for p in class_predictions if p == i) / len(class_predictions)
            print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    return accuracy


def train_nn(data_loader, args):
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    print("Загрузка тренировочных данных...")
    train_images, train_labels = data_loader.load_train_images()
    print(f"Загружено {len(train_images)} тренировочных изображений")
    
    nn = NeuralNetworkClassifier(
        base_model_name=args.nn_base_model,
        num_classes=data_loader.get_num_classes(),
        learning_rate=args.nn_learning_rate
    )
    
    nn.train(
        train_images,
        train_labels,
        epochs=args.nn_epochs,
        batch_size=args.nn_batch_size,
        use_augmentation=True
    )
    
    nn.save(args.nn_model_path)
    
    return nn


def test_nn(data_loader, args):
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    if not os.path.exists(args.nn_model_path):
        print(f"Ошибка: Модель {args.nn_model_path} не найдена!")
        return None
    
    nn = NeuralNetworkClassifier(
        base_model_name=args.nn_base_model,
        num_classes=data_loader.get_num_classes(),
        learning_rate=args.nn_learning_rate
    )
    nn.load(args.nn_model_path)
    
    print("Загрузка тестовых данных...")
    test_images, test_labels, test_paths = data_loader.load_test_images()
    print(f"Загружено {len(test_images)} тестовых изображений")
    
    if len(test_images) == 0:
        print("Тестовые данные не найдены!")
        return None
    
    accuracy = nn.evaluate(test_images, test_labels)
    print(f"\nТочность классификации (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    predictions = nn.predict(test_images)
    print("\nДетальная статистика по классам:")
    for i in range(data_loader.get_num_classes()):
        class_name = data_loader.get_class_name(i)
        class_indices = [j for j, label in enumerate(test_labels) if label == i]
        if len(class_indices) > 0:
            class_predictions = [predictions[j] for j in class_indices]
            class_accuracy = sum(1 for p in class_predictions if p == i) / len(class_predictions)
            print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    return accuracy


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.data_dir):
        print(f"Ошибка: Директория {args.data_dir} не найдена!")
        sys.exit(1)
    
    if not os.path.exists(args.train_file):
        print(f"Ошибка: Файл {args.train_file} не найден!")
        sys.exit(1)
    
    data_loader = DataLoader(args.data_dir, args.train_file)
    
    if args.algorithm in ['bow', 'both']:
        if args.mode in ['train', 'both']:
            train_bow(data_loader, args)
        
        if args.mode in ['test', 'both']:
            accuracy_bow = test_bow(data_loader, args)
            if accuracy_bow is not None:
                print(f"\nИтоговая точность Bag of Words: {accuracy_bow:.4f} ({accuracy_bow*100:.2f}%)")
    
    if args.algorithm in ['nn', 'both']:
        if args.mode in ['train', 'both']:
            train_nn(data_loader, args)
        
        if args.mode in ['test', 'both']:
            accuracy_nn = test_nn(data_loader, args)
            if accuracy_nn is not None:
                print(f"\nИтоговая точность нейронной сети: {accuracy_nn:.4f} ({accuracy_nn*100:.2f}%)")
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)


if __name__ == '__main__':
    main()
