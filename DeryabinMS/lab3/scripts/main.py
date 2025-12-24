"""
Главный скрипт для классификации достопримечательностей Нижнего Новгорода
Обеспечивает интерфейс командной строки для работы с алгоритмами
"""

import os
import sys
import argparse
import cv2

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bow_classifier import BOWClassifier
from cnn_classifier import CNNClassifier

def create_classifier(algorithm, **kwargs):
    """Фабричная функция для создания классификаторов."""
    if algorithm == 'bow':
        return BOWClassifier(
            n_clusters=kwargs.get('n_clusters', 200),
            image_size=kwargs.get('image_size', (224, 224)),
            detector_type=kwargs.get('detector_type', 'sift')
        )
    elif algorithm == 'cnn':
        return CNNClassifier(
            image_size=kwargs.get('image_size', (224, 224)),
            batch_size=kwargs.get('batch_size', 16),
            epochs=kwargs.get('epochs', 20),
            learning_rate=kwargs.get('learning_rate', 0.001),
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            base_model_name=kwargs.get('base_model_name', 'vgg16')
        )
    else:
        raise ValueError(f"Неизвестный алгоритм: {algorithm}")

def create_extended_train_file(original_train_file, extra_images_dir, data_dir):
    """Создает расширенный train файл с дополнительными изображениями."""
    print("Создание расширенной обучающей выборки...")
    
    # Читаем оригинальный train.txt
    with open(original_train_file, 'r', encoding='utf-8') as f:
        original_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Маппинг русских названий к именам папок в датасете
    folder_mapping = {
        'Нижегородский Кремль': '01_NizhnyNovgorodKremlin',
        'Дворец труда': '08_PalaceOfLabor',
        'Архангельский собор': '04_ArkhangelskCathedral'
    }
    
    # Собираем дополнительные изображения
    extra_lines = []
    for russian_name, dataset_name in folder_mapping.items():
        folder_path = os.path.join(extra_images_dir, russian_name)
        if os.path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = f"ExtDataset/{dataset_name}/{file_name}"
                    extra_lines.append(file_path)
    
    # Объединяем
    all_lines = original_lines + extra_lines
    
    # Создаем новый файл
    extended_train_file = os.path.join(data_dir, 'train_test_split', 'train_extended.txt')
    with open(extended_train_file, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(line + '\n')
    
    print(f"Оригинальных изображений: {len(original_lines)}")
    print(f"Дополнительных изображений: {len(extra_lines)}")
    print(f"Всего в расширенной выборке: {len(all_lines)}")
    print(f"Файл сохранен: {extended_train_file}")
    
    return extended_train_file

def visualize_features(image_path, output_path=None, detector_type='sift'):
    """Визуализирует ключевые точки на изображении."""
    classifier = BOWClassifier(detector_type=detector_type)
    result_image = classifier.visualize_features(image_path, output_path)
    return result_image

def test_single_image(classifier, image_path):
    """Тестирование классификатора на одном изображении."""
    print(f"\nТестирование на изображении: {image_path}")
    
    pred_label, confidence = classifier.predict_single(image_path)
    
    if pred_label:
        print(f"  Предсказанный класс: {pred_label}")
        print(f"  Уверенность: {confidence:.2%}")
        
        # Показываем изображение
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(image_rgb)
            plt.title(f'Предсказание: {pred_label} ({confidence:.1%})')
            plt.axis('off')
            plt.show()
    
    return pred_label, confidence

def main():
    parser = argparse.ArgumentParser(
        description='Классификация достопримечательностей Нижнего Новгорода'
    )
    
    # Основные параметры
    parser.add_argument('--data_dir', type=str, default='./NNClassification',
                       help='Директория с изображениями')
    parser.add_argument('--train_file', type=str, 
                       help='Файл с обучающей выборкой (если не указан, будет создан с дополнительными изображениями)')
    parser.add_argument('--test_file', type=str, default='train_test_split/test.txt',
                       help='Файл с тестовой выборкой')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'cnn'], default='bow',
                       help='bow (мешок слов) или cnn (нейронная сеть)')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both', 'visualize', 'single'],
                       default='both', help='Режим работы')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Директория для моделей')
    
    # Параметры BOW
    parser.add_argument('--clusters', type=int, default=200,
                       help='Количество кластеров для BOW')
    parser.add_argument('--detector', type=str, choices=['sift', 'orb', 'akaze'], default='sift',
                       help='Детектор для BOW')
    
    # Параметры CNN
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Размер батча для CNN')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Количество эпох для CNN')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Скорость обучения для CNN')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate для CNN')
    parser.add_argument('--base_model', type=str, default='vgg16',
                       help='Базовая модель для CNN')
    
    # Параметры изображений
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                       help='Размер изображения (ширина высота)')
    
    # Параметры визуализации
    parser.add_argument('--image_path', type=str,
                       help='Путь к изображению для визуализации')
    parser.add_argument('--output_path', type=str,
                       help='Путь для сохранения изображения')
    
    # Дополнительные параметры
    parser.add_argument('--extra_dir', type=str, default='my_images',
                       help='Директория с дополнительными изображениями')
    parser.add_argument('--plot_confusion', action='store_true',
                       help='Показывать матрицу ошибок')
    parser.add_argument('--plot_history', action='store_true',
                       help='Показывать историю обучения')
    
    args = parser.parse_args()
    
    # Визуализация
    if args.mode == 'visualize':
        if not args.image_path:
            print("Для визуализации требуется указать --image_path")
            return
        
        print(f"Визуализация {args.detector.upper()}-дескрипторов")
        visualize_features(args.image_path, args.output_path, args.detector)
        return
    
    # Обработка путей
    data_dir = args.data_dir
    extra_dir = args.extra_dir
    
    # Автоматически создаем расширенный train файл с дополнительными изображениями
    if not args.train_file:
        original_train_file = os.path.join(data_dir, 'train_test_split', 'train.txt')
        train_file = create_extended_train_file(original_train_file, extra_dir, data_dir)
    else:
        train_file = os.path.join(data_dir, args.train_file) if not os.path.isabs(args.train_file) else args.train_file
    
    test_file = os.path.join(data_dir, args.test_file) if not os.path.isabs(args.test_file) else args.test_file
    
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
    
    classifier = create_classifier(args.algorithm, **classifier_kwargs)
    
    # Режим single
    if args.mode == 'single':
        if not args.image_path:
            print("Для режима single требуется указать --image_path")
            return
        
        model_path = os.path.join(args.model_dir, f"{args.algorithm}_model")
        if os.path.exists(model_path):
            classifier.load_model(args.model_dir)
        
        test_single_image(classifier, args.image_path)
        return
    
    # Режимы train, test, both
    print("=" * 70)
    print("КЛАССИФИКАЦИЯ ДОСТОПРИМЕЧАТЕЛЬНОСТЕЙ НИЖНЕГО НОВГОРОДА")
    print("=" * 70)
    print(f"Алгоритм: {args.algorithm}")
    print(f"Режим: {args.mode}")
    print(f"Данные: {data_dir}")
    print(f"Обучающая выборка: {train_file}")
    print(f"Тестовая выборка: {test_file}")
    
    train_accuracy, test_accuracy = 0, 0
    
    # Обучение
    if args.mode in ['train', 'both']:
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)
        
        train_accuracy = classifier.train(train_file, data_dir)
        classifier.save_model(args.model_dir)
    
    # Тестирование
    if args.mode in ['test', 'both']:
        if args.mode == 'test':
            model_path = os.path.join(args.model_dir, f"{args.algorithm}_model")
            if os.path.exists(model_path):
                classifier.load_model(args.model_dir)
        
        test_accuracy = classifier.test(test_file, data_dir)
    
    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)
    
    if args.mode in ['train', 'both']:
        print(f"Accuracy на обучающей выборке: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
    
    if args.mode in ['test', 'both']:
        print(f"Accuracy на тестовой выборке: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        extra_points = 10 * test_accuracy
        print(f"Дополнительные баллы: {extra_points:.1f} из 10")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
