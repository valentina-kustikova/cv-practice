import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,
                             confusion_matrix)
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

class ImageLoader:
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def load_images_from_folder(self, folder, label):
        images, labels = [], []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(self.denoise_image(img), self.image_size)
                images.append(img)
                labels.append(label)
        return images, labels

    @staticmethod
    def denoise_image(img):
        return cv2.medianBlur(img, 3)

class FeatureExtractor:
    def __init__(self, k=100):
        self.k = k
        self.kmeans = KMeans(n_clusters=self.k, random_state=42)
        self.detector = cv2.SIFT_create()
        self.average_keypoints = 0

    def extract_features(self, images):
        descriptors_list, keypoints_counts = [], []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                keypoints_counts.append(len(keypoints))
        return descriptors_list

    def fit_kmeans(self, descriptors_list):
        all_descriptors = np.vstack([desc for desc in descriptors_list if desc is not None])
        self.kmeans.fit(all_descriptors)

    def build_histogram(self, descriptors):
        histogram = np.zeros(self.k)
        if descriptors is not None:
            predictions = self.kmeans.predict(descriptors)
            for pred in predictions:
                histogram[pred] += 1
        return histogram

    def build_feature_vectors(self, images):
        features = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.detector.detectAndCompute(gray, None)
            features.append(self.build_histogram(descriptors))
        return np.array(features)

class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        return (accuracy_score(y_test, y_pred),
                y_pred)

class Visualizer:

    @staticmethod
    def plot_confusion_matrix(cm, class_names, title):
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, str(cm[i, j]), ha="center", color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_keypoints(images, sift, number):
        for i, img in enumerate(images[0:number]):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift_keypoints, _ = sift.detectAndCompute(gray, None)
            img_with_sift_keypoints = cv2.drawKeypoints(img, sift_keypoints, None,
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(8, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(cv2.cvtColor(img_with_sift_keypoints, cv2.COLOR_BGR2RGB))
            plt.title('Keypoints')
            plt.axis('off')
            plt.show()


class ArgumentParser:
    @staticmethod
    def cli_argument_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('-td', '--train_dir', type=str, help='Path to training images')
        parser.add_argument('-tsd', '--test_dir', type=str, help='Path to testing images')
        parser.add_argument('-nc', '--n_clusters', type=int, default=100, help='Number of clusters')
        return parser.parse_args()

def main(args):
    image_loader = ImageLoader()
    feature_extractor = FeatureExtractor(k=args.n_clusters)
    model = SVC(kernel='rbf', probability=True, gamma=0.01, C=1, random_state=3)
    model_trainer = ModelTrainer(model)
    visualizer = Visualizer()

    cats_train, y_cats_train = image_loader.load_images_from_folder(os.path.join(args.train_dir, 'cats'), 0)
    dogs_train, y_dogs_train = image_loader.load_images_from_folder(os.path.join(args.train_dir, 'dogs'), 1)
    cats_test, y_cats_test = image_loader.load_images_from_folder(os.path.join(args.test_dir, 'cats'), 0)
    dogs_test, y_dogs_test = image_loader.load_images_from_folder(os.path.join(args.test_dir, 'dogs'), 1)

    X_train, y_train = shuffle(cats_train + dogs_train, y_cats_train + y_dogs_train, random_state=33)
    X_test, y_test = shuffle(cats_test + dogs_test, y_cats_test + y_dogs_test, random_state=33)

    train_descriptors = feature_extractor.extract_features(X_train)
    feature_extractor.fit_kmeans(train_descriptors)

    X_train_features = feature_extractor.build_feature_vectors(X_train)
    X_test_features = feature_extractor.build_feature_vectors(X_test)


    model_trainer.train(X_train_features, y_train)

    train_accuracy, y_train_pred = model_trainer.evaluate(X_train_features, y_train)
    test_accuracy, y_test_pred = model_trainer.evaluate(X_test_features, y_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    visualizer.show_keypoints(X_train, feature_extractor.detector, 2)
    visualizer.plot_confusion_matrix(cm_train, ['Cats', 'Dogs'], 'Train Confusion Matrix')
    visualizer.plot_confusion_matrix(cm_test, ['Cats', 'Dogs'], 'Test Confusion Matrix')

if __name__ == "__main__":
    args = ArgumentParser.cli_argument_parser()
    main(args)
