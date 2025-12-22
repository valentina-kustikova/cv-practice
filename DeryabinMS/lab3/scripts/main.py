import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к скриптам в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bow_classifier import BOWClassifier
from cnn_classifier import CNNClassifier

def create_classifier(algorithm, **kwargs):
    """Фабричная функция для создания классификаторов."""
    class_names = kwargs.get('class_names', 
                           ['Архангельский собор', 'Дворец труда', 'Нижегородский Кремль'])
    
    if algorithm.lower() == 'bow':
        return BOWClassifier(
            n_clusters=kwargs.get('n_clusters', 200),
            image_size=kwargs.get('image_size', (224, 224)),
            detector_type=kwargs.get('detector_type', 'sift'),
            class_names=class_names
        )
    elif algorithm.lower() == 'cnn':
        return CNNClassifier(
            image_size=kwargs.get('image_size', (224, 224)),
            batch_size=kwargs.get('batch_size', 16),
            epochs=kwargs.get('epochs', 20),
            learning_rate=kwargs.get('learning_rate', 0.001),
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            base_model_name=kwargs.get('base_model_name', 'vgg16'),
            class_names=class_names
        )
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

def visualize_sift_features(image_path, output_path=None, detector_type='sift'):
    """Визуализирует SIFT/ORB/AKAZE дескрипторы на изображении."""
    classifier = BOWClassifier(detector_type=detector_type)
    
    result_image = classifier.visualize_features(image_path, output_path)
    
    return result_image

def combine_train_files(original_train_file, extra_images_dir, output_file='combined_train.txt'):
    """Объединяет оригинальный train.txt с дополнительными изображениями."""
    print("Объединение оригинального train.txt с дополнительными изображениями...")
    
    # Читаем оригинальный файл
    with open(original_train_file, 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Ищем дополнительные изображения
    extra_lines = []
    
    # Маппинг названий папок
    folder_mapping = {
        'Нижегородский Кремль': '01_NizhnyNovgorodKremlin',
        'Дворец труда': '08_PalaceOfLabor',
        'Архангельский собор': '04_ArkhangelskCathedral'
    }
    
    for russian_name, dataset_name in folder_mapping.items():
        folder_path = os.path.join(extra_images_dir, russian_name)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Формируем путь в стиле исходного датасета
                    file_path = f"ExtDataset/{dataset_name}/{file_name}"
                    extra_lines.append(file_path)
    
    # Объединяем
    all_lines = original_lines + extra_lines
    
    # Записываем в новый файл
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    
    print(f"Оригинальных изображений: {len(original_lines)}")
    print(f"Дополнительных изображений: {len(extra_lines)}")
    print(f"Всего: {len(all_lines)} изображений")
    print(f"Объединенный файл сохранен как: {output_file}")
    
    return output_file, len(extra_lines)

def test_single_image(classifier, image_path):
    """Тестирует классификатор на одном изображении."""
    print(f"\nТестирование на одном изображении: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Ошибка: изображение не найдено: {image_path}")
        return
    
    # Определяем истинную метку из пути
    true_label = None
    if 'Нижегородский Кремль' in image_path or '01_NizhnyNovgorodKremlin' in image_path:
        true_label = 'Нижегородский Кремль'
    elif 'Дворец труда' in image_path or '08_PalaceOfLabor' in image_path:
        true_label = 'Дворец труда'
    elif 'Архангельский собор' in image_path or '04_ArkhangelskCathedral' in image_path:
        true_label = 'Архангельский собор'
    
    # Предсказание
    pred_label, confidence = classifier.predict_single(image_path)
    
    if pred_label:
        print(f"  Предсказанный класс: {pred_label}")
        print(f"  Уверенность: {confidence:.2%}")
        if true_label:
            print(f"  Истинный класс: {true_label}")
            print(f"  Результат: {'ВЕРНО' if pred_label == true_label else 'НЕВЕРНО'}")
        
        # Показываем изображение
        try:
            import matplotlib.pyplot as plt
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(8, 6))
                plt.imshow(image_rgb)
                plt.title(f'Предсказание: {pred_label} ({confidence:.1%})')
                if true_label:
                    plt.suptitle(f'Истинный класс: {true_label}', fontsize=10, color='green' if pred_label == true_label else 'red')
                plt.axis('off')
                plt.show()
        except:
            pass
    else:
        print("  Не удалось выполнить предсказание")

def main():
    parser = argparse.ArgumentParser(
        description='Классификация достопримечательностей Нижнего Новгорода',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Визуализация SIFT-дескрипторов
  python main.py --mode visualize --image_path NNClassification/ExtDataset/01_NizhnyNovgorodKremlin/example.jpg
  
  # Обучение и тестирование модели "Мешок слов"
  python main.py --mode both --algorithm bow --clusters 150 --detector sift
  
  # Обучение и тестирование CNN
  python main.py --mode both --algorithm cnn --epochs 10 --batch_size 8
  
  # Только тестирование предобученной модели
  python main.py --mode test --algorithm cnn --model_dir models/cnn_model
  
  # Использование дополнительных изображений
  python main.py --mode both --algorithm bow --use_extra_images --extra_dir my_images
        """
    )
    
    # Основные параметры
    parser.add_argument('--data_dir', type=str, default='./NNClassification',
                       help='Путь к директории с изображениями (по умолчанию: ./NNClassification)')
    parser.add_argument('--train_file', type=str, default='train_test_split/train.txt',
                       help='Путь к файлу со списком обучающих изображений (по умолчанию: train_test_split/train.txt)')
    parser.add_argument('--test_file', type=str, default='train_test_split/test.txt',
                       help='Путь к файлу со списком тестовых изображений (по умолчанию: train_test_split/test.txt)')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'cnn'], default='bow',
                       help='Алгоритм классификации: bow (мешок слов) или cnn (нейронная сеть)')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both', 'visualize', 'single'],
                       default='both', help='Режим работы')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Директория для сохранения/загрузки моделей (по умолчанию: models)')
    
    # Параметры для BOW
    parser.add_argument('--clusters', type=int, default=200,
                       help='Количество кластеров для метода мешок слов (по умолчанию: 200)')
    parser.add_argument('--detector', type=str, choices=['sift', 'orb', 'akaze'], default='sift',
                       help='Тип детектора для BOW: sift, orb или akaze (по умолчанию: sift)')
    
    # Параметры для CNN
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Размер батча для CNN (по умолчанию: 16)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Количество эпох для CNN (по умолчанию: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Скорость обучения для CNN (по умолчанию: 0.001)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate для CNN (по умолчанию: 0.5)')
    parser.add_argument('--base_model', type=str, choices=['vgg16', 'mobilenetv2', 'resnet50'],
                       default='vgg16', help='Базовая модель для CNN (по умолчанию: vgg16)')
    
    # Параметры изображений
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Размер изображения (ширина высота) (по умолчанию: 224 224)')
    
    # Параметры для визуализации
    parser.add_argument('--image_path', type=str,
                       help='Путь к изображению для визуализации или тестирования')
    parser.add_argument('--output_path', type=str,
                       help='Путь для сохранения изображения с визуализированными дескрипторами')
    
    # Дополнительные параметры
    parser.add_argument('--use_extra_images', action='store_true',
                       help='Использовать дополнительные изображения из my_images/')
    parser.add_argument('--extra_dir', type=str, default='my_images',
                       help='Директория с дополнительными изображениями (по умолчанию: my_images)')
    parser.add_argument('--combined_train_file', type=str, default='combined_train.txt',
                       help='Имя файла для объединенного train.txt (по умолчанию: combined_train.txt)')
    parser.add_argument('--plot_confusion', action='store_true',
                       help='Показывать матрицу ошибок')
    parser.add_argument('--plot_history', action='store_true',
                       help='Показывать историю обучения (только для CNN)')
    
    args = parser.parse_args()
    
    # Проверка обязательных параметров
    if args.mode == 'visualize' and not args.image_path:
        parser.error("Для режима visualize требуется указать --image_path")
    
    if args.mode == 'single' and not args.image_path:
        parser.error("Для режима single требуется указать --image_path")
    
    # Обработка путей
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, args.train_file) if not os.path.isabs(args.train_file) else args.train_file
    test_file = os.path.join(data_dir, args.test_file) if not os.path.isabs(args.test_file) else args.test_file
    
    # Объединение с дополнительными изображениями
    if args.use_extra_images and args.mode in ['train', 'both']:
        extra_dir = args.extra_dir
        combined_file = args.combined_train_file
        
        if os.path.exists(extra_dir):
            combined_train_file, extra_count = combine_train_files(train_file, extra_dir, combined_file)
            train_file = combined_train_file
            print(f"Используется объединенный train файл с {extra_count} дополнительными изображениями")
        else:
            print(f"Предупреждение: директория с дополнительными изображениями '{extra_dir}' не найдена")
    
    # Создание классификатора
    classifier_kwargs = {
        'n_clusters': args.clusters,
        'detector_type': args.detector,
        'image_size': tuple(args.image_size),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'base_model_name': args.base_model
    }
    
    try:
        classifier = create_classifier(args.algorithm, **classifier_kwargs)
    except ValueError as e:
        print(f"Ошибка создания классификатора: {e}")
        return
    
    # Выполнение в зависимости от режима
    if args.mode == 'visualize':
        print(f"Визуализация {args.detector.upper()}-дескрипторов для изображения: {args.image_path}")
        visualize_sift_features(args.image_path, args.output_path, args.detector)
        
    elif args.mode == 'single':
        # Загрузка модели, если она существует
        model_path = os.path.join(args.model_dir, f"{args.algorithm}_model")
        if os.path.exists(model_path):
            print(f"Загрузка модели из {model_path}")
            classifier.load_model(args.model_dir)
        
        test_single_image(classifier, args.image_path)
        
    else:
        # Обучение и/или тестирование
        print("=" * 70)
        print(f"КЛАССИФИКАЦИЯ ДОСТОПРИМЕЧАТЕЛЬНОСТЕЙ НИЖНЕГО НОВГОРОДА")
        print("=" * 70)
        print(f"Режим работы: {args.mode}")
        print(f"Алгоритм: {args.algorithm}")
        print(f"Директория с данными: {data_dir}")
        print(f"Обучающая выборка: {train_file}")
        print(f"Тестовая выборка: {test_file}")
        
        train_accuracy = 0
        test_accuracy = 0
        
        # Обучение
        if args.mode in ['train', 'both']:
            print("\n" + "=" * 60)
            print("ОБУЧЕНИЕ МОДЕЛИ")
            print("=" * 60)
            
            train_kwargs = {
                'plot_confusion': args.plot_confusion,
                'plot_history': args.plot_history,
                'validation_split': 0.2 if args.algorithm == 'cnn' else 0
            }
            
            train_accuracy = classifier.train(train_file, data_dir, **train_kwargs)
            
            # Сохранение модели
            if train_accuracy > 0:
                print(f"\nСохранение обученной модели...")
                classifier.save_model(args.model_dir)
        
        # Тестирование
        if args.mode in ['test', 'both']:
            print("\n" + "=" * 60)
            print("ТЕСТИРОВАНИЕ МОДЕЛИ")
            print("=" * 60)
            
            # Если только тестирование, загружаем модель
            if args.mode == 'test':
                model_path = os.path.join(args.model_dir, f"{args.algorithm}_model")
                if os.path.exists(model_path):
                    print(f"Загрузка модели из {model_path}")
                    if not classifier.load_model(args.model_dir):
                        print("Ошибка загрузки модели! Завершение работы.")
                        return
                else:
                    print(f"Ошибка: модель не найдена в {model_path}")
                    print("Сначала обучите модель с помощью --mode train или --mode both")
                    return
            
            test_kwargs = {
                'plot_confusion': args.plot_confusion
            }
            
            test_accuracy = classifier.test(test_file, data_dir, **test_kwargs)
        
        # Вывод итогов
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 70)
        print(f"Алгоритм: {args.algorithm}")
        print(f"Режим работы: {args.mode}")
        
        if args.mode in ['train', 'both']:
            print(f"Accuracy на обучающей выборке: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        
        if args.mode in ['test', 'both']:
            print(f"Accuracy на тестовой выборке: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
            
            # Расчет дополнительных баллов (по условиям задачи)
            extra_points = 10 * test_accuracy
            print(f"Дополнительные баллы за качество: {extra_points:.1f} из 10")
        
        print("=" * 70)

if __name__ == "__main__":
    main()
