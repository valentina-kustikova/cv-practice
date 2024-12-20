import argparse
import os
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-train', '--train',
                        help='Path to train data',
                        type=str,
                        dest='train_path')
    parser.add_argument('-test', '--test',
                        help='Path to test data',
                        type=str,
                        dest='test_path')
    parser.add_argument('-cc', '--clusters',
                        help='Count of clusters',
                        type=int,
                        dest='count_clusters')

    args = parser.parse_args()
    return args


def load_images(dataset_path, img_size=(256, 256)):
    images = []
    labels = []
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        if img_name.startswith("cat"):
            label = 1
        elif img_name.startswith("dog"):
            label = 0
        else:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


def extract_features(images):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        if descriptors is not None:
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(None)
    return keypoints_list, descriptors_list


def new_histogram(words, kmeans):
    histogram = np.zeros(kmeans.n_clusters)
    for word in words:
        histogram[word] += 1
    return histogram


def create_bow_histograms(descriptors_list, kmeans_model):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors is not None:
            words = kmeans_model.predict(descriptors)
            histogram = new_histogram(words, kmeans_model)
            histograms.append(histogram)
        else:
            histograms.append(kmeans_model.n_clusters)
    return np.array(histograms)


def plot_accuracy_histogram(true_labels, predicted_labels, class_labels):
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    class_accuracies = np.diag(confusion_mat) / confusion_mat.sum(axis=1)

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_accuracies, color='#88CFFD', edgecolor='black')
    plt.xlabel('Классы')
    plt.ylabel('Точность')
    plt.title('Точность классификации по классам')
    plt.ylim(0, 1)

    for index, accuracy in enumerate(class_accuracies):
        plt.text(index, accuracy + 0.02, f"{accuracy:.2f}", ha='center', fontsize=12, color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def visualize_keypoints(image, keypoints):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title("Keypoints")
    plt.axis('off')
    plt.show()


def main():
    args = cli_argument_parser()

    X_train, y_train = load_images(args.train_path)
    X_test, y_test = load_images(args.test_path)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    train_keypoints, train_descriptors = extract_features(X_train)

    indices = np.random.randint(0, len(X_test) - 1, size=3)
    for idx in indices:
        visualize_keypoints(X_train[idx], train_keypoints[idx])

    all_descriptors = np.vstack(train_descriptors)

    kmeans = KMeans(n_clusters=args.count_clusters, random_state=42)
    kmeans.fit(all_descriptors)

    X_train_bow = create_bow_histograms(train_descriptors, kmeans)

    _, test_descriptors = extract_features(X_test)
    X_test_bow = create_bow_histograms(test_descriptors, kmeans)

    scaler = StandardScaler()
    X_train_bow = scaler.fit_transform(X_train_bow)
    X_test_bow = scaler.transform(X_test_bow)

    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_bow, y_train)

    y_pred = model.predict(X_test_bow)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Dog", "Cat"]))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    plot_accuracy_histogram(y_test, y_pred, ['Dogs', 'Cats'])



    indices = np.random.randint(0, len(X_test) - 1, size=10)
    for idx in indices:
        plt.imshow(X_test[idx], cmap='gray')
        plt.title(f"Predicted: {'Dog' if y_pred[idx] == 0 else 'Cat'}, Actual: {'Dog' if y_test[idx] == 0 else 'Cat'}")
        plt.show()


if __name__ == '__main__':
    sys.exit(main() or 0)
