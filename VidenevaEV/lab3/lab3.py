import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1" 
import os
import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import sys

class VisualClassifier:
    def __init__(self, cluster_count=100, resize_dims=(256, 256), seed=21):
        self.resize_dims = resize_dims
        self.cluster_count = cluster_count
        self.seed = seed
        self.feature_extractor = cv2.SIFT_create()
        self.cluster_model = KMeans(n_clusters=self.cluster_count, random_state=self.seed)
        self.scaler = StandardScaler()
        self.svm = SVC(kernel='rbf', probability=True, gamma=0.001, C=10, random_state=self.seed)

    def load_data(self, directory, label):
        image_data = []
        image_labels = []
        for file in os.listdir(directory):
            img_path = os.path.join(directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, self.resize_dims)
                image_data.append(img)
                image_labels.append(label)
        return image_data, image_labels

    def process_data(self, train_dir, test_dir, class1, class2):

        train_class1, labels_train1 = self.load_data(os.path.join(train_dir, class1), 0)
        train_class2, labels_train2 = self.load_data(os.path.join(train_dir, class2), 1)
        test_class1, labels_test1 = self.load_data(os.path.join(test_dir, class1), 0)
        test_class2, labels_test2 = self.load_data(os.path.join(test_dir, class2), 1)

        X_train = train_class1 + train_class2
        y_train = labels_train1 + labels_train2
        X_test = test_class1 + test_class2
        y_test = labels_test1 + labels_test2

        X_train, y_train = shuffle(X_train, y_train, random_state=self.seed)
        X_test, y_test = shuffle(X_test, y_test, random_state=self.seed)

        return X_train, y_train, X_test, y_test

    def extract_features(self, images):
        all_descriptors = []
        for image in images:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints = self.feature_extractor.detect(gray_img, None)
            _, descriptors = self.feature_extractor.compute(gray_img, keypoints)
            all_descriptors.append(descriptors)
        return all_descriptors

    def train_model(self, X_train, y_train):
        descriptors = self.extract_features(X_train)
        combined_descriptors = np.vstack([d for d in descriptors if d is not None])
        self.cluster_model.fit(combined_descriptors)

        X_train_features = []
        for desc in descriptors:
            if desc is not None:
                clusters = self.cluster_model.predict(desc)
                feature_vector = np.bincount(clusters, minlength=self.cluster_count)
            else:
                feature_vector = np.zeros(self.cluster_count)
            X_train_features.append(feature_vector)

        X_train_features_scaled = self.scaler.fit_transform(X_train_features)
        self.svm.fit(X_train_features_scaled, y_train)

    def evaluate_model(self, X_data, y_true):
        descriptors = self.extract_features(X_data)
        X_features = []
        for desc in descriptors:
            if desc is not None:
                clusters = self.cluster_model.predict(desc)
                feature_vector = np.bincount(clusters, minlength=self.cluster_count)
            else:
                feature_vector = np.zeros(self.cluster_count)
            X_features.append(feature_vector)

        X_features_scaled = self.scaler.transform(X_features)
        predictions = self.svm.predict(X_features_scaled)

        accuracy = accuracy_score(y_true, predictions)
        print("Accuracy:", accuracy)

        return predictions

    def visualize_keypoints(self, images, count):
        plt.figure(figsize=(15, 10))
        for i in range(count):
            gray_img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            keypoints = self.feature_extractor.detect(gray_img, None)
            img_with_kp = cv2.drawKeypoints(images[i], keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.subplot(2, count, i + 1)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Original Image")

            plt.subplot(2, count, i + 1 + count)
            plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Keypoints")

        plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dir", required=True, help="Path to test data directory")
    parser.add_argument("--class1", required=True, help="First class name")
    parser.add_argument("--class2", required=True, help="Second class name")
    parser.add_argument("--clusters", type=int, default=100, help="Number of clusters for KMeans")
    parser.add_argument("--vis_images", type=int, default=3, help="Number of images to visualize")
    return parser.parse_args()


def main():
    args = get_args()

    model = VisualClassifier(cluster_count=args.clusters)

    X_train, y_train, X_test, y_test = model.process_data(
        args.train_dir, args.test_dir, args.class1, args.class2)

    model.train_model(X_train, y_train)

    print("Evaluating on training data...")
    model.evaluate_model(X_train, y_train)

    print("Evaluating on test data...")
    model.evaluate_model(X_test, y_test)

    print("Visualizing results...")
    model.visualize_keypoints(X_train, args.vis_images)


if __name__ == "__main__":
    sys.exit(main())
