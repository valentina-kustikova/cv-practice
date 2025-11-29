"""
Главный скрипт для обучения и тестирования классификаторов изображений.
"""

import argparse
import os
from pathlib import Path
from dataset_loader import DatasetLoader
from bow_classifier import BagOfWordsClassifier
from nn_classifier import NeuralNetworkClassifier
from utils import print_metrics, plot_confusion_matrix, save_results, plot_training_history, compare_algorithms


def main():
    parser = argparse.ArgumentParser(
        description='Классификация изображений достопримечательностей Нижнего Новгорода'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='NNClassification',
        help='Путь к директории с данными (по умолчанию: NNClassification)'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        default='train.txt',
        help='Путь к файлу train.txt (по умолчанию: train.txt)'
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
        help='Алгоритм: bow (мешок слов), nn (нейронная сеть), both (оба)'
    )
    
    # Параметры для BOW
    parser.add_argument(
        '--bow_vocab_size',
        type=int,
        default=300,
        help='Размер словаря для BOW (по умолчанию: 300)'
    )
    
    parser.add_argument(
        '--bow_detector',
        type=str,
        choices=['sift', 'surf', 'orb'],
        default='sift',
        help='Детектор для BOW: sift, surf, orb (по умолчанию: sift)'
    )
    
    parser.add_argument(
        '--bow_model_path',
        type=str,
        default='models/bow_model.pkl',
        help='Путь для сохранения/загрузки модели BOW'
    )
    
    # Параметры для нейросети
    parser.add_argument(
        '--nn_epochs',
        type=int,
        default=20,
        help='Количество эпох для нейросети (по умолчанию: 20)'
    )
    
    parser.add_argument(
        '--nn_batch_size',
        type=int,
        default=16,
        help='Размер батча для нейросети (по умолчанию: 16)'
    )
    
    parser.add_argument(
        '--nn_model_path',
        type=str,
        default='models/nn_model.h5',
        help='Путь для сохранения/загрузки модели нейросети'
    )
    
    parser.add_argument(
        '--model_save_dir',
        type=str,
        default='models',
        help='Директория для сохранения моделей (по умолчанию: models)'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Директория для сохранения результатов (по умолчанию: results)'
    )
    
    args = parser.parse_args()
    
    # Создаем необходимые директории
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Загружаем данные
    print("Загрузка данных...")
    loader = DatasetLoader(data_dir=args.data_dir)
    train_data, test_data = loader.split_train_test(args.train_file)
    
    class_names = [
        "Нижегородский Кремль",
        "Архангельский собор",
        "Дворец труда"
    ]
    
    # Обработка алгоритма "мешок слов"
    if args.algorithm in ['bow', 'both']:
        print("\n" + "="*70)
        print("АЛГОРИТМ 'МЕШОК СЛОВ'")
        print("="*70)
        
        bow_classifier = BagOfWordsClassifier(
            vocab_size=args.bow_vocab_size,
            detector_type=args.bow_detector
        )
        
        # Обучение
        if args.mode in ['train', 'both']:
            print("\nОбучение классификатора 'мешок слов'...")
            bow_classifier.train(train_data)
            
            # Сохранение модели
            os.makedirs(os.path.dirname(args.bow_model_path), exist_ok=True)
            bow_classifier.save(args.bow_model_path)
        else:
            # Загрузка модели
            if os.path.exists(args.bow_model_path):
                bow_classifier.load(args.bow_model_path)
            else:
                print(f"Ошибка: модель не найдена по пути {args.bow_model_path}")
                return
        
        # Тестирование
        if args.mode in ['test', 'both']:
            print("\nТестирование классификатора 'мешок слов'...")
            bow_results = bow_classifier.evaluate(test_data)
            
            print_metrics(bow_results, class_names)
            
            # Визуализация матрицы ошибок
            plot_confusion_matrix(
                bow_results['confusion_matrix'],
                class_names,
                save_path=os.path.join(args.results_dir, 'bow_confusion_matrix.png'),
                title='Матрица ошибок - Мешок слов'
            )
            
            # Сохранение результатов
            save_results(
                bow_results,
                os.path.join(args.results_dir, 'bow_results.txt'),
                'Мешок слов'
            )
    
    # Обработка нейросетевого классификатора
    if args.algorithm in ['nn', 'both']:
        print("\n" + "="*70)
        print("НЕЙРОСЕТЕВОЙ КЛАССИФИКАТОР")
        print("="*70)
        
        nn_classifier = NeuralNetworkClassifier(num_classes=3)
        
        # Обучение
        if args.mode in ['train', 'both']:
            print("\nОбучение нейросетевого классификатора...")
            nn_classifier.train(
                train_data,
                validation_data=None,  # Можно добавить валидационную выборку
                epochs=args.nn_epochs,
                batch_size=args.nn_batch_size,
                model_save_dir=args.model_save_dir
            )
            
            # Сохранение модели
            os.makedirs(os.path.dirname(args.nn_model_path), exist_ok=True)
            nn_classifier.save(args.nn_model_path)
            
            # Визуализация истории обучения
            if nn_classifier.history:
                plot_training_history(
                    nn_classifier.history,
                    save_path=os.path.join(args.results_dir, 'nn_training_history.png')
                )
        else:
            # Загрузка модели
            if os.path.exists(args.nn_model_path):
                nn_classifier.load(args.nn_model_path)
            else:
                print(f"Ошибка: модель не найдена по пути {args.nn_model_path}")
                return
        
        # Тестирование
        if args.mode in ['test', 'both']:
            print("\nТестирование нейросетевого классификатора...")
            nn_results = nn_classifier.evaluate(test_data)
            
            print_metrics(nn_results, class_names)
            
            # Визуализация матрицы ошибок
            plot_confusion_matrix(
                nn_results['confusion_matrix'],
                class_names,
                save_path=os.path.join(args.results_dir, 'nn_confusion_matrix.png'),
                title='Матрица ошибок - Нейронная сеть'
            )
            
            # Сохранение результатов
            save_results(
                nn_results,
                os.path.join(args.results_dir, 'nn_results.txt'),
                'Нейронная сеть'
            )
    
    # Сравнение алгоритмов
    if args.algorithm == 'both' and args.mode in ['test', 'both']:
        print("\n" + "="*70)
        print("СРАВНЕНИЕ АЛГОРИТМОВ")
        print("="*70)
        
        compare_algorithms(
            bow_results,
            nn_results,
            save_path=os.path.join(args.results_dir, 'algorithm_comparison.png')
        )
        
        print(f"\nAccuracy - Мешок слов: {bow_results['accuracy']:.4f}")
        print(f"Accuracy - Нейронная сеть: {nn_results['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*70)


if __name__ == '__main__':
    main()

