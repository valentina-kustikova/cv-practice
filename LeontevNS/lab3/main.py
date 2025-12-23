import argparse
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import NN
import bow
import cv2
import numpy as np
from load_data import load_images_from_split
from sklearn.model_selection import train_test_split

def cli_argument_parser():
    parser = argparse.ArgumentParser(
        description='Классификация изображений достопримечательностей Нижнего Новгорода',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обучение модели Bag of Words
  python main.py --method BOW --type train --train_file splits/train.txt --clusters 300 --output models/bow_model.joblib
  
  # Тестирование Bag of Words
  python main.py --method BOW --type test --test_file splits/test.txt --model models/bow_model.joblib
  # Обучение нейронной сети
  python main.py --method NN --type train --train_file splits/train.txt --model_name MobileNetV2
  
  # Тестирование нейронной сети
  python main.py --method NN --type test --test_file splits/test.txt --model models/mobilenet_model.keras
  
  # Визуализация дескрипторов
  python main.py --method BOW --type visualize --image data/NNSUDataset/01_NizhnyNovgorodKremlin/example.jpg
        """
    )
    
    # Основные параметры
    parser.add_argument('-m', '--method', 
                       help='Метод классификации',
                       choices=['BOW', 'NN'],
                       required=True)
    parser.add_argument('-t', '--type',
                       help='Тип операции',
                       choices=['train', 'test', 'visualize', 'predict'],
                       required=True)
    
    # Параметры данных
    parser.add_argument('--train_file', 
                       help='Путь к файлу с обучающей выборкой',
                       type=str,
                       default='splits/train.txt')
    parser.add_argument('--test_file',
                       help='Путь к файлу с тестовой выборкой',
                       type=str,
                       default='splits/test.txt')
    parser.add_argument('--val_split',
                       help='Доля валидационных данных (0.0-1.0)',
                       type=float,
                       default=0.1)
    
    # Параметры Bag of Words
    parser.add_argument('--clusters_name',
                       help='Метод кластеризации',
                       choices=['KMeans', 'MiniBatch'],
                       default='MiniBatch')
    parser.add_argument('--clf_name',
                       help='Классификатор',
                       choices=['KNN', 'SVC', 'RandomForest', 'GradientBoosting'],
                       default='SVC')
    parser.add_argument('--descriptor',
                       help='Тип дескриптора',
                       choices=['SIFT', 'ORB', 'SURF'],
                       default='SIFT')
    parser.add_argument('--k_nearest',
                       help='Количество соседей для KNN',
                       type=int,
                       default=5)
    parser.add_argument('--clusters',
                       help='Количество кластеров',
                       type=int,
                       default=300)
    parser.add_argument('--batch_size',
                       help='Размер батча для MiniBatchKMeans',
                       type=int,
                       default=1000)
    
    # Параметры нейронной сети
    parser.add_argument('--model_name', 
                       choices=["MobileNetV2", "CustomCNN"], 
                       help='Имя предобученной сверточной нейронной сети', 
                       type=str, 
                       default='MobileNetV2')
    parser.add_argument('--epochs',
                       help='Количество эпох обучения',
                       type=int,
                       default=20)
    parser.add_argument('--batch_size_nn',
                       help='Размер батча для нейронной сети',
                       type=int,
                       default=16)
    parser.add_argument('--learning_rate',
                       help='Learning rate',
                       type=float,
                       default=0.0001)
    
    # Пути моделей
    parser.add_argument('--model',
                       help='Путь к сохраненной модели',
                       type=str,
                       default=None)
    parser.add_argument('--output',
                       help='Путь для сохранения модели',
                       type=str,
                       default=None)
    
    # Параметры визуализации
    parser.add_argument('--image',
                       help='Путь к изображению для визуализации или предсказания',
                       type=str,
                       default=None)
    parser.add_argument('--visualize_histograms',
                       help='Количество гистограмм для визуализации',
                       type=int,
                       default=5)
    parser.add_argument('--save_plots',
                       help='Сохранять графики',
                       action='store_true')
    parser.add_argument('--plots_dir',
                       help='Директория для сохранения графиков',
                       type=str,
                       default='plots')
    
    # Параметры отладки
    parser.add_argument('--max_images',
                       help='Максимальное количество изображений (для отладки)',
                       type=int,
                       default=None)
    parser.add_argument('--verbose',
                       help='Подробный вывод',
                       action='store_true')
    
    return parser.parse_args()

def setup_directories():
    dirs = ['models', 'plots', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

def train_bow_model(args):
    print("=" * 60)
    print("Обучение модели Bag of Words")
    print("=" * 60)
    
    print(f"Загрузка обучающих данных из {args.train_file}...")
    train_images, train_labels, label_names = load_images_from_split(
        args.train_file, max_images=args.max_images
    )
    bow_model = bow.BOW(
        clusters_name=args.clusters_name,
        clf_name=args.clf_name,
        k_nearest=args.k_nearest,
        clusters=args.clusters,
        batch_size=args.batch_size,
        descriptor_type=args.descriptor
    )
    bow_model.extract_descriptors(train_images, verbose=args.verbose)
    bow_model.bag_of_words(verbose=args.verbose)
    bow_model.create_histograms(verbose=args.verbose)
    bow_model.train_bow_model(train_labels, verbose=args.verbose)
    if args.output:
        output_path = args.output
    else:
        output_path = f"models/bow_{args.clusters}_{args.clf_name}.joblib"
    
    bow_model.save_model(output_path)
    
    if args.save_plots:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        bow_model.plot_histograms(
            num_images=args.visualize_histograms,
            save_path=plots_dir / "bow_histograms.png"
        )
    
    print("\nОбучение завершено успешно!")
    return bow_model, label_names

def test_bow_model(args):
    print("=" * 60)
    print("Тестирование модели Bag of Words")
    print("=" * 60)

    if args.model is None:
        print("Ошибка: путь к модели не указан. Используйте --model")
        return
    
    bow_model = bow.BOW()
    bow_model.load_model(args.model)
    test_images, test_labels, label_names = load_images_from_split(
        args.test_file, max_images=args.max_images
    )
    predictions, accuracy = bow_model.test_bow_model(
        test_images, test_labels, verbose=args.verbose
    )
    if args.verbose and label_names:
        print("\nСтатистика по классам:")
        for label_id, label_name in label_names.items():
            class_indices = np.where(test_labels == label_id)[0]
            if len(class_indices) > 0:
                class_predictions = predictions[class_indices]
                class_accuracy = np.mean(class_predictions == label_id)
                print(f"  {label_name}: {len(class_indices)} изображений, точность: {class_accuracy:.3f}")
    
    print(f"\nОбщая точность: {accuracy:.4f}")
    if args.save_plots and label_names:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(test_labels, predictions)
        class_names = [label_names[i] for i in range(len(label_names))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "bow_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return accuracy, predictions

def train_nn_model(args):
    print("=" * 60)
    print("Обучение нейронной сети")
    print("=" * 60)
    
    print(f"Загрузка обучающих данных из {args.train_file}...")
    train_images, train_labels, label_names = load_images_from_split(
        args.train_file, mode="NN", max_images=args.max_images
    )
    
    print(f"\nСоздание модели {args.model_name}...")
    nn_model = NN.NN(
        model_name=args.model_name,
        input_shape=(224, 224, 3),
        num_classes=len(label_names)
    )
    
    nn_model.model_summary()
    
    if args.val_split > 0:
        print(f"\nРазделение данных: {args.val_split*100:.1f}% на валидацию")
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels,
            test_size=args.val_split,
            random_state=42,
            stratify=train_labels
        )
        print(f"  Обучающих изображений: {len(train_images)}")
        print(f"  Валидационных изображений: {len(val_images)}")
        validation_data = (val_images, val_labels)
    else:
        validation_data = None
    
    print(f"\nОбучение модели (эпох: {args.epochs}, batch size: {args.batch_size_nn})...")
    nn_model.train_nn(
        train_images=train_images,
        train_labels=train_labels,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size_nn
    )
    
    if args.output:
        model_path = args.output
    else:
        model_path = f"models/{args.model_name}_model.keras"
    
    nn_model.save(model_path)
    
    if args.save_plots:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        nn_model.plot_training_history(save_path=plots_dir / "nn_training_history.png")
    
    print("\nОбучение завершено успешно!")
    return nn_model, label_names

def test_nn_model(args):
    print("=" * 60)
    print("Тестирование нейронной сети")
    print("=" * 60)
    
    if args.model is None:
        print("Ошибка: путь к модели не указан. Используйте --model")
        return
    
    nn_model = NN.NN()
    nn_model.load(args.model)
    
    print(f"Загрузка тестовых данных из {args.test_file}...")
    test_images, test_labels, label_names = load_images_from_split(
        args.test_file, mode="NN", max_images=args.max_images
    )
    
    print("\nТестирование модели...")
    accuracy, predictions, report, cm = nn_model.test_nn(
        test_images, test_labels, batch_size=args.batch_size_nn
    )
    
    print(f"\nОбщая точность: {accuracy:.4f}")
    
    if args.verbose and label_names:
        print("\nСтатистика по классам:")
        for label_id, label_name in label_names.items():
            class_indices = np.where(test_labels == label_id)[0]
            if len(class_indices) > 0:
                class_predictions = predictions[class_indices]
                class_accuracy = np.mean(class_predictions == label_id)
                print(f"  {label_name}: {len(class_indices)} изображений, точность: {class_accuracy:.3f}")
    
    if args.save_plots and label_names and cm is not None:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        
        class_names = [label_names[i] for i in range(len(label_names))]
        nn_model.plot_confusion_matrix(
            cm, class_names,
            save_path=plots_dir / "nn_confusion_matrix.png"
        )
    
    return accuracy, predictions

def visualize_descriptors(args):
    print("=" * 60)
    print("Визуализация дескрипторов")
    print("=" * 60)
    
    if args.image is None:
        print("Ошибка: путь к изображению не указан. Используйте --image")
        return
    
    img = cv2.imread(args.image)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {args.image}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    bow_model = bow.BOW(descriptor_type=args.descriptor)
    
    print(f"\nВизуализация дескрипторов {args.descriptor}...")
    img_kp, num_kp, num_desc = bow_model.visualize_descriptors(img_rgb)
    
    print(f"Найдено {num_kp} ключевых точек и {num_desc} дескрипторов")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Оригинальное изображение")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"Дескрипторы {args.descriptor} (точек: {num_kp})")
    plt.axis('off')
    
    plt.tight_layout()
    
    if args.save_plots:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        image_name = Path(args.image).stem
        plt.savefig(plots_dir / f"descriptors_{image_name}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return img_kp

def predict_single_image(args):
    print("=" * 60)
    print("Предсказание для одного изображения")
    print("=" * 60)
    
    if args.image is None:
        print("Ошибка: путь к изображению не указан. Используйте --image")
        return
    
    if args.model is None:
        print("Ошибка: путь к модели не указан. Используйте --model")
        return
    
    img = cv2.imread(args.image)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {args.image}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if args.method == "BOW":
        bow_model = bow.BOW()
        bow_model.load_model(args.model)
        
        predictions = bow_model.test_bow_model([img_rgb], verbose=False)
        predicted_label = predictions[0]
        
        
    elif args.method == "NN":
        nn_model = NN.NN()
        nn_model.load(args.model)
        
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        predicted_label, probabilities = nn_model.predict([img_resized])
        predicted_label = predicted_label[0]
        probabilities = probabilities[0]
        
        print(f"\nВероятности по классам:")
        for i, prob in enumerate(probabilities):
            print(f"  Класс {i}: {prob:.4f}")
    
    print(f"\nПредсказанный класс: {predicted_label}")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Предсказанный класс: {predicted_label}")
    plt.axis('off')
    
    if args.save_plots:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(exist_ok=True)
        image_name = Path(args.image).stem
        plt.savefig(plots_dir / f"prediction_{image_name}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return predicted_label

def main():
    args = cli_argument_parser()
    
    setup_directories()
    
    if args.type == 'train':
        if args.method == 'BOW':
            train_bow_model(args)
        elif args.method == 'NN':
            train_nn_model(args)
    
    elif args.type == 'test':
        if args.method == 'BOW':
            test_bow_model(args)
        elif args.method == 'NN':
            test_nn_model(args)
    
    elif args.type == 'visualize':
        visualize_descriptors(args)
    
    elif args.type == 'predict':
        predict_single_image(args)
    
    else:
        print(f"Неизвестный тип операции: {args.type}")

if __name__ == "__main__":
    main()