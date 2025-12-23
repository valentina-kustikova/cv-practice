import argparse
import os
import sys
from pathlib import Path
from data_loader import DataLoader
from bag_of_words import BagOfWords


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Классификация изображений достопримечательностей Нижнего Новгорода с помощью Bag of Words'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.',
        help='Путь до директории с данными (по умолчанию: текущая директория)'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        default='train.txt',
        help='Путь к файлу со списком тренировочных изображений (по умолчанию: train.txt)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Режим работы: train (обучение), test (тестирование), both (обучение и тестирование)'
    )
    
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=200,
        help='Размер словаря для Bag of Words (по умолчанию: 200)'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        choices=['SIFT', 'ORB', 'AKAZE'],
        default='SIFT',
        help='Тип детектора для Bag of Words (по умолчанию: SIFT)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='bow_model.pkl',
        help='Путь для сохранения/загрузки модели Bag of Words'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Создавать визуализации ключевых точек'
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
        vocab_size=args.vocab_size,
        detector_type=args.detector,
        descriptor_type=args.detector,
        use_tfidf=True
    )
    
    bow.train(train_images, train_labels, visualize=args.visualize)
    
    bow.save(args.model_path)
    
    return bow


def test_bow(data_loader, args):
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ BAG OF WORDS")
    print("="*60)
    
    if not os.path.exists(args.model_path):
        print(f"Ошибка: Модель {args.model_path} не найдена!")
        return None
    
    bow = BagOfWords(
        vocab_size=args.vocab_size,
        detector_type=args.detector,
        descriptor_type=args.detector,
        use_tfidf=True
    )
    bow.load(args.model_path)
    
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


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.data_dir):
        print(f"Ошибка: Директория {args.data_dir} не найдена!")
        sys.exit(1)
    
    if not os.path.exists(args.train_file):
        print(f"Ошибка: Файл {args.train_file} не найден!")
        sys.exit(1)
    
    data_loader = DataLoader(args.data_dir, args.train_file)
    
    if args.mode in ['train', 'both']:
        train_bow(data_loader, args)
    
    if args.mode in ['test', 'both']:
        accuracy_bow = test_bow(data_loader, args)
        if accuracy_bow is not None:
            print(f"\nИтоговая точность Bag of Words: {accuracy_bow:.4f} ({accuracy_bow*100:.2f}%)")
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)


if __name__ == '__main__':
    main()

