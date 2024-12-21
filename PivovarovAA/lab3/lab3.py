import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from sklearn.utils import shuffle

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img)
            labels.append(label)
    return images, labels

class DataframeExtractor:
    def __init__(self, n_clusters=100, descriptor='sift'):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        if descriptor == 'sift':
            self.detector = cv2.SIFT_create()
        elif descriptor == 'orb':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    def extract_features(self, images):
        descriptors_list = []
        keypoints_counts = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                keypoints_counts.append(len(keypoints))
        self.average_keypoints = np.mean(keypoints_counts) if keypoints_counts else 0
        return descriptors_list

    def compute_bow_histograms(self, images):
        features = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            histogram = np.zeros(self.n_clusters)
            if descriptors is not None:
                predictions = self.kmeans.predict(descriptors)
                for pred in predictions:
                    histogram[pred] += 1
            features.append(histogram)
        return np.array(features)

    def fit_kmeans(self, descriptors_list):
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None:
                all_descriptors.append(desc)
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans.fit(all_descriptors)

class ModelClass:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', min_samples_split=4, min_samples_leaf=3, max_depth=4, max_features='log2', random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def stats(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred, y_prob

def visualize_features(images, descriptor):
    for i, img in enumerate(images[:5]):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints = descriptor.detect(gray, None)
            img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(6, 6))
            plt.imshow(img_with_keypoints, cmap='gray')
            plt.title(f"Image {i + 1}: {len(keypoints)} keypoints detected")
            plt.axis('off')
            plt.show()
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='coolwarm')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_descriptors_histogram(cats_descriptors, dogs_descriptors, kmeans, n_clusters):
    all_cats_descriptors = np.vstack([desc for desc in cats_descriptors if desc is not None])
    all_dogs_descriptors = np.vstack([desc for desc in dogs_descriptors if desc is not None])

    cats_predictions = kmeans.predict(all_cats_descriptors)
    dogs_predictions = kmeans.predict(all_dogs_descriptors)

    cats_histogram = np.bincount(cats_predictions, minlength=n_clusters)
    dogs_histogram = np.bincount(dogs_predictions, minlength=n_clusters)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.bar(range(n_clusters), cats_histogram, color='skyblue')
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Descriptors')
    plt.title(f'Histogram of Descriptors Distribution for Cats')

    plt.subplot(1, 2, 2)
    plt.bar(range(n_clusters), dogs_histogram, color='orange')
    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Descriptors')
    plt.title(f'Histogram of Descriptors Distribution for Dogs')

    plt.tight_layout()
    plt.show()

def plot_classification_results(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    cats_count = cm[0, 0].sum()
    dogs_count = cm[1, 1].sum()

    misclassified_count = cm[0, 1] + cm[1, 0]

    counts = [cats_count, dogs_count, misclassified_count]
    labels = ['Cats', 'Dogs', 'Misclassified']
    colors = ['green', 'green', 'red']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color=colors)

    plt.xlabel('Categories')
    plt.ylabel('Number of images')
    plt.title('Classification Results')

    for i, count in enumerate(counts):
        plt.text(i, count + 1, str(count), ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()


class ArgumentParser:
    def cli_argument_parser():
        parser = argparse.ArgumentParser(description="Image classification using various algorithms and Bag of Words with SIFT and ORB.")

        parser.add_argument('-td', '--train_dir',
                            help='Directory with training images (cats and dogs)',
                            type=str,
                            dest='train_dir')
        parser.add_argument('-tsd', '--test_dir',
                            help='Directory with test images',
                            type=str,
                            dest='test_dir')
        parser.add_argument('-nc', '--n_clusters',
                            help='Number of clusters for visual dictionary',
                            type=int,
                            dest='n_clusters',
                            default=100)
        parser.add_argument('-d', '--descriptor',
                            help='Descriptor to use sift or orb',
                            type=str,
                            choices=['sift', 'orb'],
                            dest='descriptor',
                            default='sift')

        args = parser.parse_args()
        return args

def main(args):
    #python lab3.py -td dataset\train -tsd dataset\test -nc 100 -d sift
    
    data = DataframeExtractor(n_clusters=args.n_clusters, descriptor=args.descriptor)

    model = ModelClass()

    cats_train, y_cats_train = load_images(os.path.join(args.train_dir, 'cats'), 0)
    dogs_train, y_dogs_train = load_images(os.path.join(args.train_dir, 'dogs'), 1)
    cats_test, y_cats_test = load_images(os.path.join(args.test_dir, 'cats'), 0)
    dogs_test, y_dogs_test = load_images(os.path.join(args.test_dir, 'dogs'), 1)

    Train_images = cats_train + dogs_train
    Train_labels = y_cats_train + y_dogs_train
    Test_images = cats_test + dogs_test
    Test_labels = y_cats_test + y_dogs_test

    Train_images, Train_labels = shuffle(Train_images, Train_labels, random_state=42)
    Test_images, Test_labels = shuffle(Test_images, Test_labels, random_state=42)
    
    
    train_descriptors = data.extract_features(Train_images)
    data.fit_kmeans(train_descriptors)
    print(f"Среднее количество точек, обнаруженных на тренировочных изображениях: {data.average_keypoints}")
    
    print(f"Визуализация ключевых точек для некоторых картинок...")
    visualize_features(Train_images, data.detector)
    
    print(f"Визуализация дескрипторов для двух классов...")
    cats_descriptors = data.extract_features([cats_train[10]])
    dogs_descriptors = data.extract_features([dogs_train[10]])
    plot_descriptors_histogram(cats_descriptors, dogs_descriptors, data.kmeans, data.n_clusters)
    
    print(f"Преобразование дескрипторов в гистограммы частот...")
    X_train_features = data.compute_bow_histograms(Train_images)
    X_test_features = data.compute_bow_histograms(Test_images)
    
    print(f"Тренировка модели...")
    model.train(X_train_features, Train_labels)

    train_accuracy, y_train_pred, y_train_prob = model.stats(X_train_features, Train_labels)
    test_accuracy, y_test_pred, y_test_prob = model.stats(X_test_features, Test_labels)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)


    class_names = ['Cats', 'Dogs']
    plot_confusion_matrix(Test_labels, y_test_pred, class_names)
    plot_classification_results(Test_labels, y_test_pred, class_names)
    
if __name__ == "__main__":
    args = ArgumentParser.cli_argument_parser()
    main(args)
