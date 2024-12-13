#Импортируем необходимые библиотеки
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not find the number of physical cores.*")

#Функция для разбора аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Bag of Visual Words Image Classifier")
    parser.add_argument('-trd', '--train_dir', type=str, required=True, help='Path to the training data directory')
    parser.add_argument('-tsd', '--test_dir', type=str, required=True, help='Path to the testing data directory')
    parser.add_argument('-ncl', '--num_clusters', type=int, default=100, help='Number of clusters for KMeans')
    return parser.parse_args()

#Реализация алгоритма "мешок слов"
class BoVWFeatureExtractor:
    def __init__(self, num_clusters=100):
        self.num_clusters = num_clusters
        self.sift = cv2.SIFT_create()
        self.kmeans = None

    #Извлечение признаков изображений
    def load_images_from_folder(self, folder, label, image_size=(256, 256)):
        images = []
        labels = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
        return images, labels

    #Извлечение SIFT дескрипторов 
    def extract_sift_features(self, images):
        print("Извлечение SIFT дескрипторов")
        descriptors_list = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.sift.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
        return descriptors_list

    #Обучение модели KMeans
    def train_kmeans(self, descriptors_list):
        print("Обучение KMeans")
        all_descriptors = np.vstack([desc for desc in descriptors_list if desc is not None])
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

    #Создание гистограммы 
    def build_histogram(self, descriptors):
        histogram = np.zeros(self.num_clusters)
        if descriptors is not None:
            predictions = self.kmeans.predict(descriptors)
            for pred in predictions:
                histogram[pred] += 1
        return histogram

    #Построение вектора признаков 
    def build_feature_vectors(self, images):
        print("Построение векторов признаков")
        features = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.sift.detectAndCompute(gray, None)
            histogram = self.build_histogram(descriptors)
            features.append(histogram)
        return np.array(features)

    #Визуализация ключевых точек
    def visualize_features(self, images, num_images=5):
        import random 
    
        random_indices = random.sample(range(len(images)), min(num_images, len(images)))
        
        for i, idx in enumerate(random_indices):
            img = images[idx]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift_keypoints, _ = self.sift.detectAndCompute(gray, None)
            img_with_sift_keypoints = cv2.drawKeypoints(
                img, sift_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
    
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(img_with_sift_keypoints, cv2.COLOR_BGR2RGB))
            plt.title(f'SIFT Keypoints - Image {idx + 1}')
            plt.axis('off')
            plt.show()


    #Визуализация гистограммы для каждого класса
    def visualize_combined_histogram(self, descriptors_list, labels, class_names):
        print("Визуализация комбинированных гистограмм")
        combined_histograms = {label: np.zeros(self.num_clusters) for label in np.unique(labels)}

        for descriptors, label in zip(descriptors_list, labels):
            if descriptors is not None:
                histogram = self.build_histogram(descriptors)
                combined_histograms[label] += histogram

        for label, histogram in combined_histograms.items():
            plt.bar(range(self.num_clusters), histogram, alpha=0.7, label=class_names[label])

        plt.xlabel('Кластеры')
        plt.ylabel('Частота')
        plt.title('Комбинированная гистограмма кластеров дескрипторов')
        plt.legend()
        plt.show()

#Классификация изображений 
class BoVWClassifier:
    def __init__(self, feature_extractor, train_data, test_data):
        self.feature_extractor = feature_extractor
        self.X_train, self.y_train = train_data
        self.X_test, self.y_test = test_data
        self.scaler = StandardScaler()
        self.clf = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=42)

    def train(self):
        #Извлечение SIFT дескрипторов
        train_descriptors = self.feature_extractor.extract_sift_features(self.X_train)
        self.feature_extractor.train_kmeans(train_descriptors)

        #Построение векторов признаков
        X_train_features = self.feature_extractor.build_feature_vectors(self.X_train)
        X_test_features = self.feature_extractor.build_feature_vectors(self.X_test)

        # Нормализация веторов признаков
        X_train_features = self.scaler.fit_transform(X_train_features)
        X_test_features = self.scaler.transform(X_test_features)

        #Обучение SVM классификатора
        print("Обучение SVM классификатора")
        self.clf.fit(X_train_features, self.y_train)

        #Оценка на train
        y_train_pred = self.clf.predict(X_train_features)
        print("Точность на train:", accuracy_score(self.y_train, y_train_pred))
        print("Отчёт train:")
        print(classification_report(self.y_train, y_train_pred))

        #Оценка на test
        y_test_pred = self.clf.predict(X_test_features)
        print("Точность на test:", accuracy_score(self.y_test, y_test_pred))
        print("Отчёт test:")
        print(classification_report(self.y_test, y_test_pred))

        #Гистограмма классификации
        self.plot_classification_histogram(self.y_train, y_train_pred, ['Cats', 'Dogs'])
        self.plot_classification_histogram(self.y_test, y_test_pred, ['Cats', 'Dogs'])

        #Матрица ошибок
        self.plot_confusion_matrix(self.y_test, y_test_pred, ['Cats', 'Dogs'])

    #Графики точности классификации по классам
    def plot_classification_histogram(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        correct_counts = np.diag(cm)
        total_counts = np.sum(cm, axis=1)
        accuracy_per_class = correct_counts / total_counts

        plt.figure(figsize=(10, 6))
        plt.bar(class_names, accuracy_per_class, color='#CCCCFF')
        plt.xlabel('Классы')
        plt.ylabel('Точность')
        plt.title('График точности')
        plt.ylim(0, 1)

        for i, acc in enumerate(accuracy_per_class):
            plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', va='bottom', fontsize=12)

        plt.show()

    #Отображение матрицы ошибок
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Матрица ошибок')
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.ylabel('Истина')
        plt.xlabel('Предсказание')
        plt.show()

#Основная функция
if __name__ == "__main__":
    args = parse_args()

    #Извлечение признаков из изображений
    feature_extractor = BoVWFeatureExtractor(num_clusters=args.num_clusters)

    #Загрузка данных для обучения и тестирования
    train_cats, y_train_cats = feature_extractor.load_images_from_folder(os.path.join(args.train_dir, 'cats'), 0)
    train_dogs, y_train_dogs = feature_extractor.load_images_from_folder(os.path.join(args.train_dir, 'dogs'), 1)
    test_cats, y_test_cats = feature_extractor.load_images_from_folder(os.path.join(args.test_dir, 'cats'), 0)
    test_dogs, y_test_dogs = feature_extractor.load_images_from_folder(os.path.join(args.test_dir, 'dogs'), 1)

    #Объединение данных
    train_data = (train_cats + train_dogs, y_train_cats + y_train_dogs)
    test_data = (test_cats + test_dogs, y_test_cats + y_test_dogs)

    #Запуск классификатора
    classifier = BoVWClassifier(feature_extractor, train_data, test_data)
    classifier.train()

    #Ключевые точки
    feature_extractor.visualize_features(train_cats + train_dogs)

    #Гистограмма распределения признаков
    all_train_descriptors = feature_extractor.extract_sift_features(train_cats + train_dogs)
    labels = y_train_cats + y_train_dogs
    feature_extractor.visualize_combined_histogram(all_train_descriptors, labels, ['Cats', 'Dogs'])
