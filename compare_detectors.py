"""
Скрипт для сравнения различных детекторов и дескрипторов
для алгоритма Bag of Words
"""
import argparse
import os
import sys
from pathlib import Path
from data_loader import DataLoader
from bag_of_words import BagOfWords


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Сравнение различных детекторов для Bag of Words'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.',
        help='Путь до директории с данными'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        default='train.txt',
        help='Путь к файлу со списком тренировочных изображений'
    )
    
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=200,
        help='Размер словаря для Bag of Words'
    )
    
    parser.add_argument(
        '--detectors',
        type=str,
        nargs='+',
        choices=['SIFT', 'ORB', 'AKAZE'],
        default=['SIFT', 'ORB', 'AKAZE'],
        help='Список детекторов для сравнения'
    )
    
    return parser.parse_args()


def evaluate_detector(data_loader, detector_type, vocab_size):
    """Оценка качества классификации для конкретного детектора"""
    print(f"\n{'='*60}")
    print(f"ОЦЕНКА ДЕТЕКТОРА: {detector_type}")
    print(f"{'='*60}")
    
    print("Загрузка тренировочных данных...")
    train_images, train_labels = data_loader.load_train_images()
    print(f"Загружено {len(train_images)} тренировочных изображений")
    
    print("Загрузка тестовых данных...")
    test_images, test_labels, _ = data_loader.load_test_images()
    print(f"Загружено {len(test_images)} тестовых изображений")
    
    if len(test_images) == 0:
        print("Тестовые данные не найдены!")
        return None
    
    # Создание и обучение модели
    bow = BagOfWords(
        vocab_size=vocab_size,
        detector_type=detector_type,
        descriptor_type=detector_type,
        use_tfidf=True
    )
    
    print(f"\nОбучение модели с детектором {detector_type}...")
    bow.train(train_images, train_labels, visualize=False)
    
    # Тестирование
    accuracy = bow.evaluate(test_images, test_labels)
    print(f"\nТочность классификации: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Детальная статистика по классам
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
    
    results = {}
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ДЕТЕКТОРОВ ДЛЯ BAG OF WORDS")
    print("="*60)
    print(f"Размер словаря: {args.vocab_size}")
    print(f"Детекторы для сравнения: {', '.join(args.detectors)}")
    
    for detector in args.detectors:
        try:
            accuracy = evaluate_detector(data_loader, detector, args.vocab_size)
            if accuracy is not None:
                results[detector] = accuracy
        except Exception as e:
            print(f"\nОшибка при оценке детектора {detector}: {e}")
            results[detector] = None
    
    # Итоговая таблица результатов
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"{'Детектор':<15} {'Точность':<15} {'Процент':<15}")
    print("-" * 45)
    
    for detector, accuracy in sorted(results.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True):
        if accuracy is not None:
            print(f"{detector:<15} {accuracy:<15.4f} {accuracy*100:<15.2f}%")
        else:
            print(f"{detector:<15} {'Ошибка':<15} {'-':<15}")
    
    # Определение лучшего детектора
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_detector = max(valid_results, key=valid_results.get)
        best_accuracy = valid_results[best_detector]
        print(f"\nЛучший детектор: {best_detector} (точность: {best_accuracy:.4f} = {best_accuracy*100:.2f}%)")
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)


if __name__ == '__main__':
    main()

