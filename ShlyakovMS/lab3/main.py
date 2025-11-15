import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import numpy as np
import torch
from src.utils import visualize_bow_beautifully 
from src.dataset import load_split_file, load_images_from_list
from src.bow_classifier import BoWClassifier
from src.cnn_classifier import TransferLearningClassifier
from sklearn.metrics import classification_report, accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Landmark Classification — BoW + CNN")
    parser.add_argument('--data_dir', type=str, default='data', help='Путь к папке data')
    parser.add_argument('--train_split', type=str, default='data/train.txt')
    parser.add_argument('--test_split', type=str, default='data/test.txt')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both')
    parser.add_argument('--k', type=int, default=400, help='Размер словаря BoW')
    parser.add_argument('--epochs', type=int, default=15, help='Эпох для CNN')
    args = parser.parse_args()

    # === ЗАГРУЗКА ДАННЫХ ===
    print("Загрузка разбиения...")
    train_files = load_split_file(args.train_split)
    test_files = load_split_file(args.test_split)
    print("Загрузка изображений...")
    train_images, train_labels = load_images_from_list(train_files, args.data_dir)
    test_images, test_labels = load_images_from_list(test_files, args.data_dir)
    print(f"Готово: {len(train_images)} train, {len(test_images)} test")

    # === 1. BoW + SVM ===
    print("\n" + "═" * 70)
    print(f" 1. ЗАПУСК BoW + SIFT + SVM (k={args.k})")
    print("═" * 70)

    bow = BoWClassifier(k=args.k)
    pred_bow = None
    acc_bow = None

    if args.mode in ['train', 'both']:
        print(" → Извлечение дескрипторов и обучение...")
        desc_list, _ = bow.extract_sift(train_images)
        bow.train(train_images, train_labels, desc_list)
        bow.save("bow_model.pkl")
        print(" Модель BoW сохранена")

    if args.mode in ['test', 'both']:
        if os.path.exists("bow_model.pkl"):
            bow.load("bow_model.pkl")
            pred_bow, _ = bow.predict(test_images)
            acc_bow = accuracy_score(test_labels, pred_bow)
            print(f"\n BoW + SVM → Accuracy: {acc_bow:.4%}")
            print(classification_report(test_labels, pred_bow,
                                      target_names=['Кремль', 'Дворец труда', 'Архангельский собор'],
                                      digits=4))
            with open("bow_result.txt", "w", encoding="utf-8") as f:
                f.write(f"BoW Accuracy: {acc_bow:.4%}\n\n")
                f.write(classification_report(test_labels, pred_bow,
                                            target_names=['Кремль', 'Дворец труда', 'Архангельский собор']))
        else:
            print(" bow_model.pkl не найден!")

    if args.mode in ['test', 'both'] and pred_bow is not None:
        print("\n" + "═" * 70)
        print(" КРАСИВАЯ ВИЗУАЛИЗАЦИЯ BoW")
        print("═" * 70)
        bow_vis = BoWClassifier(k=args.k)
        bow_vis.load("bow_model.pkl")
        visualize_bow_beautifully(test_images, test_labels, pred_bow, bow_vis, top_k=5)

    print("\n" + "═" * 70)
    print(" 2. ЗАПУСК ResNet50 (Transfer Learning)")
    print("═" * 70)

    cnn = TransferLearningClassifier(num_classes=3)
    pred_cnn = None
    acc_cnn = None

    if args.mode in ['train', 'both']:
        print(f" → Обучение ({args.epochs} эпох)...")
        cnn.train(train_images, train_labels, epochs=args.epochs)
        cnn.save("cnn_model.pth")
        print(" Модель CNN сохранена")

    if args.mode in ['test', 'both']:
        if os.path.exists("cnn_model.pth"):
            cnn.model.load_state_dict(torch.load("cnn_model.pth", map_location=cnn.device))
            pred_cnn, _ = cnn.predict(test_images)
            acc_cnn = accuracy_score(test_labels, pred_cnn)
            print(f"\n ResNet50 → Accuracy: {acc_cnn:.4%}")
            print(classification_report(test_labels, pred_cnn,
                                      target_names=['Кремль', 'Дворец труда', 'Архангельский собор'],
                                      digits=4))
            with open("cnn_result.txt", "w", encoding="utf-8") as f:
                f.write(f"CNN Accuracy: {acc_cnn:.4%}\n\n")
                f.write(classification_report(test_labels, pred_cnn,
                                            target_names=['Кремль', 'Дворец труда', 'Архангельский собор']))
        else:
            print(" cnn_model.pth не найден!")

    print("\n" + "=" * 70)
    print(" ФИНАЛЬНОЕ СРАВНЕНИЕ")
    print("=" * 70)
    if acc_bow is not None and acc_cnn is not None:
        print(f" BoW + SVM: {acc_bow:.4%}")
        print(f" ResNet50 CNN: {acc_cnn:.4%}")
        print(f" Превосходство CNN: +{(acc_cnn - acc_bow)*100:.2f}%")
    elif acc_bow is not None:
        print(f" BoW + SVM: {acc_bow:.4%}")
    elif acc_cnn is not None:
        print(f" ResNet50 CNN: {acc_cnn:.4%}")

    print("\n" + "="*60)
    print(" ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ!")
    print(" Нажмите Enter, чтобы выйти...")
    print("="*60)
    input()


if __name__ == "__main__":
    main()