import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Bag of Visual Words Image Classifier")

    parser.add_argument('-trd', '--train_data', 
                        type=str,
                        dest='train_data',
                        required=True, 
                        help='Path to train data')

    parser.add_argument('-tsd', '--test_data',
                        type=str, 
                        dest='test_data', 
                        required=True, 
                        help='Path to test data')

    parser.add_argument('-tsl', '--test_labels', 
                        type=str, 
                        dest='test_labels', 
                        required=True, 
                        help='Path to test labels')

    parser.add_argument('-d', '--descriptor',
                        help='Descriptor to use sift or orb',
                        type=str,
                        dest='descriptor',
                        choices=['sift', 'orb'],
                        default='sift')

    parser.add_argument('-a', '--algorithm',
                            help='Algorithm to use for training and testing',
                            type=str,
                            choices=['svm', 'knn', 'rf'],
                            dest='algorithm',
                            default='svm')

    parser.add_argument('-c', '--count_clusters', 
                        type=int, 
                        dest='count_clusters',
                        default=50, 
                        help='Count of clusters')

    return parser.parse_args()

def load_train_dataset(train_path, num_samples_per_class=100):
    images, labels = [], []
    
    for label, category in enumerate(["cat", "dog"]):
        image_files = [f for f in os.listdir(train_path) if category in f][:num_samples_per_class]

        for file in image_files:
            img_path = os.path.join(train_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

def load_test_dataset(test_path, labels_file):
    images, labels = [], []

    # Загрузка меток
    with open(labels_file, "r") as file:
        lines = file.readlines()
        label_map = {line.split()[0]: int(line.split()[1]) for line in lines}

    print(f"Загружено меток: {len(label_map)}")
    
    for file in os.listdir(test_path):
        img_path = os.path.join(test_path, file)
        
        file_key = os.path.splitext(file)[0]
        if file_key not in label_map:
            print(f"Файл {file} не найден в label_map.")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Ошибка загрузки изображения: {img_path}")
            continue

        img = cv2.resize(img, (256, 256))
        images.append(img)
        labels.append(label_map[file_key])

    print(f"Загружено изображений: {len(images)}")
    return np.array(images), np.array(labels)

class Classifier:
    def __init__(self, algorithm='svm'):
        self.scaler = StandardScaler()
        if algorithm == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
        elif algorithm == 'svm':
            self.model = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=42)
        elif algorithm == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', min_samples_split=4, min_samples_leaf=3, max_depth=3, max_features='log2', random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def stats(self, X, y):
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        accuracy = accuracy_score(y, y_pred)
        return accuracy, y_pred, y_prob

class ExtractorFeatures:
    def __init__(self, num_clusters=50, descriptor='sift'):
        self.num_clusters = num_clusters
        self.kmeans = None

        if descriptor == 'sift':
            self.extractor = cv2.SIFT_create()
        elif descriptor == 'orb':
            self.extractor = cv2.ORB_create()
        else:
            raise ValueError(f'Unsupported descriptor: {descriptor}')

    def extract_features(self, images):
        descriptors_list = []
        for img in images:
            keypoints, descriptors = self.extractor.detectAndCompute(img, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
            else:
                descriptors_list.append(None)
        return descriptors_list
    
    def compute_histograms(self, images):
        features = []
        for img in images:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.extractor.detectAndCompute(img, None)
            histogram = self.build_histogram(descriptors)
            features.append(histogram)
        return np.array(features)

    def create_bow(self, descriptors_list):
        all_descriptors = []
        for desc in descriptors_list:
            if desc is not None:
                all_descriptors.append(desc)
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

    def build_histogram(self, descriptors):
        histogram = np.zeros(self.num_clusters)
        if descriptors is not None:
            predictions = self.kmeans.predict(descriptors)
            for pred in predictions:
                histogram[pred] += 1
        return histogram
    
    def visualize_descriptors_histogram(self, descriptors_list, labels, class_names):
        print("Визуализация гистограмм кластеров дескрипторов")
        combined_histograms = {label: np.zeros(self.num_clusters) for label in np.unique(labels)}

        for descriptors, label in zip(descriptors_list, labels):
            if descriptors is not None:
                histogram = self.build_histogram(descriptors)
                combined_histograms[label] += histogram

        num_classes = len(combined_histograms)
        fig, axes = plt.subplots(1, num_classes, figsize=(10, 5), sharey=True)

        for ax, (label, histogram) in zip(axes, combined_histograms.items()):
            ax.bar(range(self.num_clusters), histogram, alpha=0.7)
            ax.set_xlabel('Cluster number')
            ax.set_title(f'{class_names[label]}')
    
        axes[0].set_ylabel('Number of Descriptors')
        plt.suptitle('Histograms of descriptor clusters')
        plt.tight_layout()
        plt.show()

def visualize_results(y_test, y_pred, labels):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, [np.sum(y_test == 0), np.sum(y_test == 1)], alpha=0.7, label="True")
    plt.bar(labels, [np.sum(y_pred == 0), np.sum(y_pred == 1)], alpha=0.7, label="Predicted", width=0.4)
    plt.title("True vs Predicted Class Distribution")
    plt.ylabel("Number of Samples")
    plt.legend()
    plt.show()

def visualize_features(images, descriptor):
    for i, img in enumerate(images[:1]):
        keypoints = descriptor.detect(img, None)
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f"Image {i + 1}: {len(keypoints)} keypoints detected")
        plt.axis('off')
        plt.show()

def visualize_confusion_matrix(y_true, y_pred, class_names):
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

def main():
    args = parse_args()

    # train_path = "dogs-vs-cats/train"
    # test_path = "dogs-vs-cats/test"
    # labels_file = "dogs-vs-cats/test_labels.txt"

    class_names = ['Cats', 'Dogs']
    train_path = args.train_data
    test_path = args.test_data
    labels_file = args.test_labels
    num_samples = 100
    count_clusters = args.count_clusters

    X_train, y_train = load_train_dataset(train_path, num_samples)
    X_test, y_test = load_test_dataset(test_path, labels_file)
    print(f"Train dataset size: {X_test.shape}")

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    feature_extractor = ExtractorFeatures(num_clusters=count_clusters, descriptor=args.descriptor)
    train_descriptors = feature_extractor.extract_features(X_train)
    test_descriptors = feature_extractor.extract_features(X_test)

    # Создание "мешка слов"
    feature_extractor.create_bow(train_descriptors)

    visualize_features(X_train, feature_extractor.extractor)

    feature_extractor.visualize_descriptors_histogram(train_descriptors, y_train, class_names)

    train_histograms = feature_extractor.compute_histograms(X_train)
    test_histograms = feature_extractor.compute_histograms(X_test)

    # print(f"test_histograms shape: {test_histograms.shape}")
    # print(f"train_histograms shape: {train_histograms.shape}")

    classifier = Classifier(algorithm=args.algorithm)
    classifier.train(train_histograms, y_train)

    train_accuracy, y_train_pred, y_train_prob = classifier.stats(train_histograms, y_train)
    test_accuracy, y_test_pred, y_test_prob = classifier.stats(test_histograms, y_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names = class_names))

    visualize_results(y_test, y_test_pred, class_names)
    visualize_confusion_matrix(y_test, y_test_pred, class_names)

if __name__ == "__main__":
    main()