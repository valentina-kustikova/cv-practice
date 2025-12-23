import argparse
import os
import sys
from pathlib import Path
from data_loader import DataLoader

# Условный импорт для избежания проблем с TensorFlow
try:
    from neural_network import NeuralNetworkClassifier
    TF_AVAILABLE = True
except Exception as e:
    print(f"Предупреждение: Не удалось импортировать TensorFlow: {e}")
    print("Попробуйте обновить зависимости: pip install tensorflow numpy<2.0.0")
    TF_AVAILABLE = False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Классификация изображений достопримечательностей Нижнего Новгорода с помощью нейронной сети'
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
        '--base_model',
        type=str,
        choices=['ResNet50', 'VGG16', 'MobileNetV2'],
        default='ResNet50',
        help='Базовая модель для transfer learning (по умолчанию: ResNet50)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Количество эпох обучения (по умолчанию: 30, рекомендуется не более 50)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Размер батча (по умолчанию: 32, больше = меньше нагрузка)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Скорость обучения (по умолчанию: 0.0001)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='nn_model.h5',
        help='Путь для сохранения/загрузки модели'
    )
    
    return parser.parse_args()


def train_nn(data_loader, args):
    if not TF_AVAILABLE:
        print("Ошибка: TensorFlow недоступен. Невозможно обучить нейронную сеть.")
        return None
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    print("Загрузка тренировочных данных...")
    train_images, train_labels = data_loader.load_train_images()
    print(f"Загружено {len(train_images)} тренировочных изображений")
    
    nn = NeuralNetworkClassifier(
        base_model_name=args.base_model,
        num_classes=data_loader.get_num_classes(),
        learning_rate=args.learning_rate
    )
    
    nn.train(
        train_images,
        train_labels,
        epochs=min(args.epochs, 50),  # Ограничение максимум 50 эпох
        batch_size=max(args.batch_size, 16),  # Минимум batch size 16
        use_augmentation=True,
        use_class_weights=True,
        fine_tune_after=None  # Отключен для экономии ресурсов
    )
    
    nn.save(args.model_path)
    
    return nn


def test_nn(data_loader, args):
    if not TF_AVAILABLE:
        print("Ошибка: TensorFlow недоступен. Невозможно протестировать нейронную сеть.")
        return None
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    if not os.path.exists(args.model_path):
        print(f"Ошибка: Модель {args.model_path} не найдена!")
        return None
    
    nn = NeuralNetworkClassifier(
        base_model_name=args.base_model,
        num_classes=data_loader.get_num_classes(),
        learning_rate=args.learning_rate
    )
    nn.load(args.model_path)
    
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
    
    if not TF_AVAILABLE:
        print("\nОшибка: TensorFlow недоступен.")
        print("Попробуйте установить совместимые версии:")
        print("  pip install 'tensorflow>=2.13.0' 'numpy<2.0.0,>=1.24.0'")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        print(f"Ошибка: Директория {args.data_dir} не найдена!")
        sys.exit(1)
    
    if not os.path.exists(args.train_file):
        print(f"Ошибка: Файл {args.train_file} не найден!")
        sys.exit(1)
    
    data_loader = DataLoader(args.data_dir, args.train_file)
    
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

