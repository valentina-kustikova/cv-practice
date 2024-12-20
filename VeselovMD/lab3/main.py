import os
import zipfile
import random

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def load_images_from_folder(folder, label, image_size=(150, 150)):
    """
    Загружает изображения из указанной папки, изменяет их размер и возвращает массив изображений и меток.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels


def visualize_histogram(hist, title="Гистограмма слов"):
    """
    Визуализация гистограммы частот.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(hist)), hist)
    plt.title(title)
    plt.xlabel("Индекс визуального слова")
    plt.ylabel("Частота")
    plt.show()


def extract_features_bag_of_words(images, n_clusters=50):
    """
    Извлечение признаков с помощью алгоритма "мешок слов".
    """
    sift = cv2.SIFT_create()
    descriptors = []

    # Извлекаем дескрипторы для всех изображений
    for img in images:
        keypoints, descriptor = sift.detectAndCompute(img, None)
        if descriptor is not None:
            descriptors.append(descriptor)

    # Проверка на наличие дескрипторов
    if len(descriptors) == 0:
        print("Ошибка: не удалось извлечь дескрипторы ни для одного изображения.")
        return np.array([]), None

    # Объединяем все дескрипторы
    descriptors = np.vstack(descriptors)

    # Кластеризация с использованием MiniBatchKMeans
    print("Кластеризация дескрипторов...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors)

    # Построение гистограмм
    histograms = []
    for img in images:
        keypoints, descriptor = sift.detectAndCompute(img, None)
        histogram = np.zeros(n_clusters, dtype=np.float32)
        if descriptor is not None:
            words = kmeans.predict(descriptor)
            for word in words:
                histogram[word] += 1
        histograms.append(histogram)

    return np.array(histograms), kmeans


def main():
    # Параметры
    image_size = (150, 150)
    train_folder_cats = "dataset/train/cats"
    train_folder_dogs = "dataset/train/dogs"
    test_folder_cats = "dataset/test/cats"
    test_folder_dogs = "dataset/test/dogs"
    n_clusters = 50

    # Загружаем тренировочные и тестовые данные
    train_cats, train_labels_cats = load_images_from_folder(train_folder_cats, "cat", image_size)
    train_dogs, train_labels_dogs = load_images_from_folder(train_folder_dogs, "dog", image_size)
    test_cats, test_labels_cats = load_images_from_folder(test_folder_cats, "cat", image_size)
    test_dogs, test_labels_dogs = load_images_from_folder(test_folder_dogs, "dog", image_size)

    train_images = train_cats + train_dogs
    train_labels = train_labels_cats + train_labels_dogs
    test_images = test_cats + test_dogs
    test_labels = test_labels_cats + test_labels_dogs

    # Извлечение признаков для тренировочной выборки
    print("Извлечение признаков с использованием алгоритма 'мешок слов'...")
    train_histograms, kmeans = extract_features_bag_of_words(train_images, n_clusters)

    # Извлечение признаков для тестовой выборки
    test_histograms = []
    sift = cv2.SIFT_create()
    for img in test_images:
        keypoints, descriptor = sift.detectAndCompute(img, None)
        histogram = np.zeros(n_clusters, dtype=np.float32)
        if descriptor is not None:
            words = kmeans.predict(descriptor)
            for word in words:
                histogram[word] += 1
        test_histograms.append(histogram)
    test_histograms = np.array(test_histograms)

    # Преобразуем метки в числовой формат
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Обучаем классификатор
    print("Обучение модели классификации...")
    model = SVC(kernel="linear")
    model.fit(train_histograms, train_labels_encoded)

    # Оцениваем модель
    print("Оценка модели на тестовой выборке...")
    test_predictions = model.predict(test_histograms)
    accuracy = accuracy_score(test_labels_encoded, test_predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_))

    # Визуализация гистограммы одного изображения
    visualize_histogram(train_histograms[0], title="Гистограмма слов для первого изображения (train)")

    # Вывод 5 случайных изображений с предсказаниями и матрицей точек
    print("Вывод 5 случайных изображений с ключевыми точками и предсказаниями...")
    random_indices = random.sample(range(len(test_images)), 5)
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(random_indices):
        # Получаем исходное изображение и ключевые точки
        img = test_images[idx]
        keypoints, _ = sift.detectAndCompute(img, None)
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_with_keypoints = img_with_keypoints[:, :, ::-1]  # Преобразуем BGR в RGB для matplotlib

        # Предсказываем класс
        predicted_label = label_encoder.inverse_transform([test_predictions[idx]])[0]
        true_label = label_encoder.inverse_transform([test_labels_encoded[idx]])[0]

        # Отображаем изображение
        plt.subplot(1, 5, i + 1)
        plt.imshow(img_with_keypoints)
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
