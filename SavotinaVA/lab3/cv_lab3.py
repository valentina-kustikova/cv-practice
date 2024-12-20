import cv2
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys


class ImageClassifier:
    def __init__(self, k=100, image_size=(256, 256), random_state=21):
        self.image_size = image_size
        self.k = k
        self.random_state = random_state
        self.detector = cv2.SIFT_create()
        self.descriptor = cv2.SIFT_create()
        self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        self.scaler = StandardScaler()
        self.X_test_features = None
        self.clf = SVC(kernel='rbf', probability=True, gamma=0.001, C=10, random_state=21)
        # Лучшие параметры: {'clf': SVC(), 'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}

    def load_images_from_folder(self, folder, label):
        images = []
        labels = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, self.image_size)
                images.append(img)
                labels.append(label)
        return images, labels

    def prepare_data(self, train_folder, test_folder, first_class_name, second_class_name):
        print("Загрузка изображений")
        cats_train, y_cats_train = self.load_images_from_folder(os.path.join(train_folder, first_class_name), 0)
        dogs_train, y_dogs_train = self.load_images_from_folder(os.path.join(train_folder, second_class_name), 1)
        cats_test, y_cats_test = self.load_images_from_folder(os.path.join(test_folder, first_class_name), 0)
        dogs_test, y_dogs_test = self.load_images_from_folder(os.path.join(test_folder, second_class_name), 1)

        X_train = cats_train + dogs_train
        y_train = y_cats_train + y_dogs_train
        X_test = cats_test + dogs_test
        y_test = y_cats_test + y_dogs_test

        X_train, y_train = shuffle(X_train, y_train, random_state=21)
        X_test, y_test = shuffle(X_test, y_test, random_state=21)

        return X_train, y_train, X_test, y_test

    def extract_descriptors(self, images):
        descriptors_list = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            key_points = self.detector.detect(gray, None)
            _, descriptors = self.descriptor.compute(gray, key_points)
            descriptors_list.append(descriptors)
        return descriptors_list

    def build_feature_vectors(self, descriptors_list):
        features = []
        for descriptors in descriptors_list:
            histogram = np.zeros(self.k)
            if descriptors is not None:
                predictions = self.kmeans.predict(descriptors)
                for pred in predictions:
                    histogram[pred] += 1
            features.append(histogram)
        return np.array(features)

    def train(self, X_train, y_train):
        train_descriptors = self.extract_descriptors(X_train)
        all_descriptors = np.vstack([desc for desc in train_descriptors if desc is not None])
        self.kmeans.fit(all_descriptors)
        X_train_features = self.build_feature_vectors(train_descriptors)
        X_train_features = self.scaler.fit_transform(X_train_features)
        self.clf.fit(X_train_features, y_train)

    def evaluate(self, X_test, y_test):
        test_descriptors = self.extract_descriptors(X_test)
        self.X_test_features = self.build_feature_vectors(test_descriptors)
        self.X_test_features = self.scaler.transform(self.X_test_features)
        y_pred = self.clf.predict(self.X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        print("Точность:", accuracy)
        print("Результаты:")
        print(classification_report(y_test, y_pred))
        return y_pred

    def plot_classification_histogram(self, class_names, y_train_true, y_train_pred, y_test_true, y_test_pred):
        cm_train = confusion_matrix(y_train_true, y_train_pred)
        cm_test = confusion_matrix(y_test_true, y_test_pred)

        correct_counts_train = np.diag(cm_train)
        total_counts_train = np.sum(cm_train, axis=1)
        accuracy_per_class_train = correct_counts_train / total_counts_train

        correct_counts_test = np.diag(cm_test)
        total_counts_test = np.sum(cm_test, axis=1)
        accuracy_per_class_test = correct_counts_test / total_counts_test

        x = np.arange(len(class_names))
        width = 0.35

        plt.figure(figsize=(10, 7))

        plt.bar(x - width / 2, accuracy_per_class_train, width, label='Тренировочная выборка', color='turquoise')
        plt.bar(x + width / 2, accuracy_per_class_test, width, label='Тестовая выборка', color='coral')

        plt.xlabel('Классы')
        plt.ylabel('Точность')
        plt.title('Сравнение точности классификации для тренировочной и тестовой выборок')
        plt.xticks(x, class_names)
        plt.ylim(0, 1)
        plt.legend()

        for i, (train_acc, test_acc) in enumerate(zip(accuracy_per_class_train, accuracy_per_class_test)):
            plt.text(i - width / 2, train_acc + 0.02, f"{train_acc:.2f}", ha='center', va='bottom', fontsize=12)
            plt.text(i + width / 2, test_acc + 0.02, f"{test_acc:.2f}", ha='center', va='bottom', fontsize=12)

        plt.show()

    def visualize_features(self, images, image_num):
        image_num = min(image_num, len(images))
        plt.figure(figsize=(12, 8))

        for i in range(image_num):
            gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            key_points = self.detector.detect(gray, None)
            img_with_keypoints = cv2.drawKeypoints(images[i], key_points, None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.subplot(2, image_num, i + 1)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.title(f'Оригинал {i + 1}')
            plt.axis('off')

            plt.subplot(2, image_num, i + 1 + image_num)
            plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
            plt.title(f'Дескрипторы {i + 1}')
            plt.axis('off')

        plt.show()
        
    def plot_confusion_matrix(self, class_names, y_train_true, y_train_pred, y_test_true, y_test_pred):
        cm_train = confusion_matrix(y_train_true, y_train_pred)
        cm_test = confusion_matrix(y_test_true, y_test_pred)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
        axes[0].set_title('Матрица ошибок на тренировочной выборке')
        disp_train.plot(ax=axes[0], cmap='RdPu', colorbar=False)
        axes[0].set_xlabel("Предсказанные метки")
        axes[0].set_ylabel("Истинные метки")

        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
        axes[1].set_title('Матрица ошибок на тестовой выборке')
        disp_test.plot(ax=axes[1], cmap='RdPu', colorbar=False)
        axes[1].set_xlabel("Предсказанные метки")
        axes[1].set_ylabel("Истинные метки")

        fig.suptitle('Сравнение матриц ошибок для тренировочных и тестовых наборов', fontsize=16)

        plt.show()

    def visualize_results(self, class_names, X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, image_num):
        print("Построение гистограммы точности для тренировочного и тестового набора данных")
        self.plot_classification_histogram(class_names, y_train, y_train_pred, y_test, y_test_pred)

        print(f"Визуализация дескрипторов для {image_num} изображений")
        self.visualize_features(X_train, image_num)

        print("Построение матриц ошибок для тренировочной и тестовой выборок")
        self.plot_confusion_matrix(class_names, y_train, y_train_pred, y_test, y_test_pred)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train",
                        help = "Path to source train directory",
                        required = True)
    parser.add_argument("-te", "--test",
                        help = "Path to source test directory",
                        required = True)
    parser.add_argument("-cn1", "--first_class_name",
                        help = "First class name",
                        required = True)
    parser.add_argument("-cn2", "--second_class_name",
                        help = "Second class name",
                        required = True)
    parser.add_argument("-k", "--clusters",
                        help = "Number of clusters")
    parser.add_argument("-in", "--image_number",
                        help = "Number of images",
                        type = int,
                        default = 3)
    return parser.parse_args()


def main():
    args = parse_arguments()
    class_names = ["Кошки", "Собаки"]

    classifier = ImageClassifier(int(args.clusters))

    print("Подготовка данных")
    X_train, y_train, X_test, y_test = classifier.prepare_data(
        args.train, args.test, args.first_class_name, args.second_class_name)

    print("Обучение модели")
    classifier.train(X_train, y_train)
    print("Оценка модели на тренировочной выборке")
    y_train_pred = classifier.evaluate(X_train, y_train)
    print("Оценка модели на тестовой выборке")
    y_test_pred = classifier.evaluate(X_test, y_test)
    print("Визуализация результатов")
    classifier.visualize_results(class_names, X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, args.image_number)

if __name__ == '__main__':
    sys.exit(main() or 0)