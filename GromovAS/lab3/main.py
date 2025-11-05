import argparse
import os
import cv2
import numpy as np
from bow_classifier import BOWClassifier
from nn_classifier import NNClassifier
from utils import load_dataset_split, load_images_from_split


def main():
    parser = argparse.ArgumentParser(description='Классификация изображений достопримечательностей НН')
    parser.add_argument('--data_dir', type=str, required=True, help='Путь к директории с данными')
    parser.add_argument('--train_split', type=str, required=True, help='Файл с тренировочной выборкой')
    parser.add_argument('--test_split', type=str, required=True, help='Файл с тестовой выборкой')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], required=True,
                        help='Режим работы: train, test или both')
    parser.add_argument('--algorithm', type=str, choices=['bow', 'nn'], required=True,
                        help='Алгоритм: bow (мешок слов) или nn (нейронная сеть)')
    parser.add_argument('--model_path', type=str, default='model.pkl',
                        help='Путь для сохранения/загрузки модели')

    # Параметры для BOW
    parser.add_argument('--vocab_size', type=int, default=1000,
                        help='Размер словаря для BOW')
    parser.add_argument('--detector', type=str, default='SIFT',
                        help='Детектор для BOW (SIFT, ORB, SURF)')

    # Параметры для нейронной сети
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох для обучения нейронной сети')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча для нейронной сети')

    args = parser.parse_args()

    # Загрузка данных
    print("Загрузка данных...")
    train_files, train_labels = load_dataset_split(args.train_split)
    test_files, test_labels = load_dataset_split(args.test_split)

    # Загрузка изображений
    train_images = load_images_from_split(args.data_dir, train_files, train_labels)
    test_images = load_images_from_split(args.data_dir, test_files, test_labels)

    # Инициализация классификатора
    if args.algorithm == 'bow':
        classifier = BOWClassifier(vocab_size=args.vocab_size, detector=args.detector)
    else:
        classifier = NNClassifier(epochs=args.epochs, batch_size=args.batch_size)

    # Обучение и/или тестирование
    if args.mode in ['train', 'both']:
        print("Обучение классификатора...")
        classifier.train(train_images, args.model_path)

    if args.mode in ['test', 'both']:
        print("Тестирование классификатора...")
        accuracy, report = classifier.test(test_images, args.model_path)
        print(f"Точность: {accuracy:.4f}")
        print("Отчет по классификации:")
        print(report)


if __name__ == "__main__":
    main()