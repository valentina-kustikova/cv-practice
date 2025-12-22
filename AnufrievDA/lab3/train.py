import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.dataset_loader import load_split_lists
from models.bovw_model import BoVWClassifier
from models.cnn_model import CNNClassifier

def main():
    parser = argparse.ArgumentParser(description="Lab 3: Image Classification")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--train_file', type=str, default='train.txt', help='Path to train split file')
    parser.add_argument('--algo', type=str, choices=['bovw', 'cnn'], default='bovw', help='Algorithm type')
    
    # НОВЫЕ АРГУМЕНТЫ: Режим работы и путь к файлу модели
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--model_path', type=str, default='saved_model', help='Path to save/load the model')

    # Параметры BoVW
    parser.add_argument('--detector', type=str, default='SIFT', choices=['SIFT', 'ORB', 'AKAZE'])
    parser.add_argument('--clusters', type=int, default=100, help='Number of visual words')
    parser.add_argument('--vis_kp', action='store_true', help='Visualize keypoints for one image')
    
    # Параметры CNN
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    
    # 1. Загрузка данных
    print("Loading data...")
    train_paths, test_paths, train_labels, test_labels = load_split_lists(args.train_file, args.data_dir)
    
    # Определяем расширение файла модели
    if args.algo == 'bovw':
        model_file = args.model_path + ".pkl"
    else:
        model_file = args.model_path + ".pth"

    model = None

    # --- РЕЖИМ ОБУЧЕНИЯ ---
    if args.mode == 'train':
        if len(train_paths) == 0:
            print("Error: No training data found.")
            return

        print(f"--- Training Mode ({args.algo}) ---")
        
        # Инициализация
        if args.algo == 'bovw':
            model = BoVWClassifier(n_clusters=args.clusters, detector_name=args.detector)
            if args.vis_kp and len(train_paths) > 0:
                model.visualize_keypoints(train_paths[0], save_to="vis_keypoints.jpg")
        elif args.algo == 'cnn':
            model = CNNClassifier(model_name=args.model, num_classes=3)

        # Обучение
        if args.algo == 'bovw':
            model.fit(train_paths, train_labels)
        else:
            model.fit(train_paths, train_labels, epochs=args.epochs, batch_size=args.batch_size)
        
        # Сохранение
        model.save(model_file)
        print(f"Model saved to {model_file}")
        
        # Сразу проверим на тесте (для удобства), но это опционально
        print("\nEvaluating on test set after training...")
        preds = model.predict(test_paths)

    # --- РЕЖИМ ТЕСТИРОВАНИЯ ---
    elif args.mode == 'test':
        print(f"--- Testing Mode ({args.algo}) ---")
        
        if not os.path.exists(model_file):
            print(f"Error: Model file '{model_file}' not found. Run --mode train first.")
            return

        # Загрузка
        print(f"Loading model from {model_file}...")
        if args.algo == 'bovw':
            model = BoVWClassifier.load(model_file)
        elif args.algo == 'cnn':
            model = CNNClassifier.load(model_file)
            
        # Предсказание
        print("Predicting on test set...")
        preds = model.predict(test_paths)

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    if preds is not None and len(test_labels) > 0:
        acc = accuracy_score(test_labels, preds)
        cm = confusion_matrix(test_labels, preds)
        
        print("\n" + "="*30)
        print(f"RESULTS")
        print("="*30)
        print(f"Accuracy: {acc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(test_labels, preds, target_names=['Kremlin', 'Cathedral', 'Palace']))

if __name__ == "__main__":
    main()