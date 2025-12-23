"""
Скрипт для исследования влияния размера словаря на качество классификации
"""
import argparse
import os
import sys
from pathlib import Path
from data_loader import DataLoader
from bag_of_words import BagOfWords


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Исследование влияния размера словаря на качество Bag of Words'
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
        '--detector',
        type=str,
        choices=['SIFT', 'ORB', 'AKAZE'],
        default='SIFT',
        help='Тип детектора для использования'
    )
    
    parser.add_argument(
        '--vocab_sizes',
        type=int,
        nargs='+',
        default=[100, 150, 200, 250, 300],
        help='Размеры словаря для исследования'
    )
    
    return parser.parse_args()


def evaluate_vocab_size(data_loader, detector_type, vocab_size):
    """Оценка качества классификации для конкретного размера словаря"""
    print(f"\nОценка размера словаря: {vocab_size}")
    print("-" * 60)
    
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
    
    print(f"Обучение модели со словарем размером {vocab_size}...")
    bow.train(train_images, train_labels, visualize=False)
    
    # Тестирование
    accuracy = bow.evaluate(test_images, test_labels)
    print(f"Точность классификации: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
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
    print("ИССЛЕДОВАНИЕ ВЛИЯНИЯ РАЗМЕРА СЛОВАРЯ")
    print("="*60)
    print(f"Детектор: {args.detector}")
    print(f"Размеры словаря для исследования: {args.vocab_sizes}")
    
    for vocab_size in sorted(args.vocab_sizes):
        try:
            accuracy = evaluate_vocab_size(data_loader, args.detector, vocab_size)
            if accuracy is not None:
                results[vocab_size] = accuracy
        except Exception as e:
            print(f"\nОшибка при оценке размера словаря {vocab_size}: {e}")
            results[vocab_size] = None
    
    # Итоговая таблица результатов
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"{'Размер словаря':<20} {'Точность':<15} {'Процент':<15}")
    print("-" * 50)
    
    for vocab_size, accuracy in sorted(results.items()):
        if accuracy is not None:
            print(f"{vocab_size:<20} {accuracy:<15.4f} {accuracy*100:<15.2f}%")
        else:
            print(f"{vocab_size:<20} {'Ошибка':<15} {'-':<15}")
    
    # Определение лучшего размера словаря
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_vocab_size = max(valid_results, key=valid_results.get)
        best_accuracy = valid_results[best_vocab_size]
        print(f"\nЛучший размер словаря: {best_vocab_size} (точность: {best_accuracy:.4f} = {best_accuracy*100:.2f}%)")
        
        # Анализ тренда
        sorted_sizes = sorted(valid_results.keys())
        if len(sorted_sizes) > 1:
            print("\nАнализ результатов:")
            for i in range(len(sorted_sizes) - 1):
                curr_size = sorted_sizes[i]
                next_size = sorted_sizes[i + 1]
                curr_acc = valid_results[curr_size]
                next_acc = valid_results[next_size]
                diff = next_acc - curr_acc
                if diff > 0:
                    print(f"  Увеличение с {curr_size} до {next_size}: +{diff:.4f} (+{diff*100:.2f}%)")
                elif diff < 0:
                    print(f"  Увеличение с {curr_size} до {next_size}: {diff:.4f} ({diff*100:.2f}%)")
                else:
                    print(f"  Увеличение с {curr_size} до {next_size}: без изменений")
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)


if __name__ == '__main__':
    main()

